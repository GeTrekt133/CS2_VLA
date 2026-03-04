"""
GPU-based embedding cache for efficient sequence encoding.

Aligned with final_model/ architecture:
  - Scene cache: emb_dim=512 (YOLO embed, NOT 2048)
  - Radar cache: emb_dim=512
  - Audio cache: StereoAudioEncoder → (1, 32, 512) → linspace (1, 16, 512)
    Re-encode when 0.5 sec of new data (8000 stereo samples)
"""

import torch
import numpy as np
from typing import Optional, Callable


class GPUEmbeddingCache:
    """
    GPU ring buffer for caching embeddings.

    Stores embeddings on GPU to avoid:
    - Re-encoding duplicate frames (63/64 are duplicates!)
    - CPU↔GPU memory transfers

    Memory layout:
    - self.buffer: torch.Tensor (capacity, emb_dim) on GPU
    - self.timestamps: np.ndarray (capacity,) on CPU
    """

    def __init__(
        self,
        capacity: int,
        emb_dim: int,
        device: str = 'cuda',
        enable_validation: bool = False
    ):
        """
        Args:
            capacity: Maximum number of embeddings to cache
            emb_dim: Embedding dimension (512 for both radar and scene)
            device: GPU device
            enable_validation: Whether to hash frames for cache validation
        """
        self.capacity = capacity
        self.emb_dim = emb_dim
        self.device = device
        self.enable_validation = enable_validation

        self.buffer = torch.zeros(
            capacity, emb_dim,
            dtype=torch.float32,
            device=device
        )

        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self.head = 0
        self.count = 0

        if enable_validation:
            self.frame_hashes = np.zeros(capacity, dtype=np.uint64)
        else:
            self.frame_hashes = None

    def add_and_get_sequence(
        self,
        new_frame: torch.Tensor,
        encoder: Callable[[torch.Tensor], torch.Tensor],
        seq_length: int,
        timestamp: Optional[float] = None
    ) -> torch.Tensor:
        """
        Encode new frame, add to cache, and return sequence of embeddings.

        Args:
            new_frame: (1, C, H, W) new frame to encode
            encoder: Encoder function → (1, emb_dim) or (emb_dim,)
            seq_length: Length of sequence to return
            timestamp: Optional timestamp

        Returns:
            embeddings: (seq_length, emb_dim) sequence on GPU
        """
        # 1. Encode the new frame
        with torch.no_grad():
            new_emb = encoder(new_frame)

            if new_emb.dim() == 3:
                new_emb = new_emb.squeeze(1)
            if new_emb.dim() == 2:
                new_emb = new_emb.squeeze(0)
            # now (emb_dim,)

        # 2. Store in ring buffer
        self.buffer[self.head] = new_emb

        if timestamp is not None:
            self.timestamps[self.head] = timestamp

        if self.frame_hashes is not None:
            self.frame_hashes[self.head] = self._hash_frame(new_frame)

        # 3. Update pointers
        self.head = (self.head + 1) % self.capacity
        self.count += 1

        # 4. Retrieve sequence of last seq_length embeddings
        if self.count < seq_length:
            valid = min(self.count, seq_length)
            padding = seq_length - valid

            if padding > 0:
                if self.head >= valid:
                    indices = torch.arange(self.head - valid, self.head, device=self.device)
                else:
                    indices = torch.cat([
                        torch.arange(self.capacity - (valid - self.head), self.capacity, device=self.device),
                        torch.arange(0, self.head, device=self.device)
                    ])

                valid_embeds = self.buffer[indices]
                padded = torch.cat([
                    torch.zeros(padding, self.emb_dim, device=self.device, dtype=torch.float32),
                    valid_embeds
                ], dim=0)
                return padded
            else:
                indices = torch.arange(self.head - valid, self.head, device=self.device)
                return self.buffer[indices]

        # Normal case: retrieve last seq_length embeddings
        if self.head >= seq_length:
            indices = torch.arange(self.head - seq_length, self.head, device=self.device)
        else:
            indices = torch.cat([
                torch.arange(self.capacity + self.head - seq_length, self.capacity, device=self.device),
                torch.arange(0, self.head, device=self.device)
            ])

        return self.buffer[indices]

    def add_embedding(self, embedding: torch.Tensor, timestamp: Optional[float] = None):
        """
        Add a pre-computed embedding to the cache (no encoding).

        Args:
            embedding: (emb_dim,) or (1, emb_dim) embedding tensor
        """
        if embedding.dim() == 2:
            embedding = embedding.squeeze(0)

        self.buffer[self.head] = embedding

        if timestamp is not None:
            self.timestamps[self.head] = timestamp

        self.head = (self.head + 1) % self.capacity
        self.count += 1

    def get_sequence(self, seq_length: int) -> torch.Tensor:
        """
        Get last seq_length embeddings without adding new ones.

        Returns:
            (seq_length, emb_dim) on GPU
        """
        if self.count == 0:
            return torch.zeros(seq_length, self.emb_dim, device=self.device)

        valid = min(self.count, seq_length)
        padding = seq_length - valid

        if self.head >= valid:
            indices = torch.arange(self.head - valid, self.head, device=self.device)
        else:
            indices = torch.cat([
                torch.arange(self.capacity - (valid - self.head), self.capacity, device=self.device),
                torch.arange(0, self.head, device=self.device)
            ])

        result = self.buffer[indices]
        if padding > 0:
            result = torch.cat([
                torch.zeros(padding, self.emb_dim, device=self.device),
                result
            ], dim=0)
        return result

    def invalidate(self):
        """Clear cache (e.g., on round restart)."""
        self.head = 0
        self.count = 0
        self.buffer.zero_()
        self.timestamps.fill(0)
        if self.frame_hashes is not None:
            self.frame_hashes.fill(0)

    def get_stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "count": self.count,
            "head": self.head,
            "filled_ratio": min(self.count / self.capacity, 1.0),
            "memory_mb": (self.capacity * self.emb_dim * 4) / (1024 * 1024),
        }

    def _hash_frame(self, frame: torch.Tensor) -> np.uint64:
        mean = frame.mean().item()
        std = frame.std().item()
        hash_val = int((mean * 1e6 + std * 1e6)) % (2**64)
        return np.uint64(hash_val)


class AudioEmbeddingCache:
    """
    Cache for stereo audio embeddings to avoid re-encoding.

    StereoAudioEncoder processes full 16-sec stereo window → 32 embeddings → linspace 16.
    Re-encode only when 0.5 sec of new audio data accumulated (8000 samples per channel).
    """

    def __init__(
        self,
        embedding_duration_ms: int = 500,
        sample_rate: int = 16000,
        device: str = 'cuda'
    ):
        self.cached_embeddings: Optional[torch.Tensor] = None  # (1, 16, 512) after linspace
        self.last_encoded_position = 0
        self.embedding_step_samples = int(embedding_duration_ms / 1000 * sample_rate)  # 8000
        self.device = device

        self.cache_hits = 0
        self.cache_misses = 0

    def get_embeddings(
        self,
        audio_tensor: torch.Tensor,
        current_position: int,
        encoder: Callable[[torch.Tensor], torch.Tensor],
        preprocessor=None,
    ) -> torch.Tensor:
        """
        Get audio embeddings, using cache if possible.

        Args:
            audio_tensor: (1, 2, 256000) stereo audio
            current_position: Current sample position in audio stream
            encoder: StereoAudioEncoder model
            preprocessor: Preprocessor with encode_audio() for linspace

        Returns:
            embeddings: (1, 16, 512) audio embeddings (linspaced from 32)
        """
        samples_since_encode = current_position - self.last_encoded_position

        if samples_since_encode < self.embedding_step_samples and self.cached_embeddings is not None:
            self.cache_hits += 1
            return self.cached_embeddings

        self.cache_misses += 1

        with torch.no_grad():
            raw_embeds = encoder(audio_tensor)  # (1, 32, 512)

        # Linspace 32 → 16
        if raw_embeds.shape[1] > 16:
            import torch.nn.functional as F
            embeds = raw_embeds.permute(0, 2, 1)  # (1, 512, 32)
            embeds = F.interpolate(embeds, size=16, mode='linear', align_corners=True)
            embeds = embeds.permute(0, 2, 1)  # (1, 16, 512)
        else:
            embeds = raw_embeds

        self.cached_embeddings = embeds
        self.last_encoded_position = current_position

        return embeds

    def invalidate(self):
        self.cached_embeddings = None
        self.last_encoded_position = 0

    def get_stats(self) -> dict:
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "memory_mb": 0.03 if self.cached_embeddings is not None else 0.0,  # 16*512*4 bytes
        }
