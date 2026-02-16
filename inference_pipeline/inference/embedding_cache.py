"""
GPU-based embedding cache for efficient sequence encoding.

Key features:
- All embeddings stored in GPU VRAM (no CPU↔GPU transfers)
- Ring buffer design for constant memory usage
- Frame-level caching with timestamp tracking
- Cache invalidation on game state changes
"""

import torch
import numpy as np
from typing import Optional, Callable, Tuple


class GPUEmbeddingCache:
    """
    GPU ring buffer for caching embeddings.

    Stores embeddings on GPU to avoid:
    - Re-encoding duplicate frames (31/32 are duplicates!)
    - CPU↔GPU memory transfers

    Memory layout:
    - self.buffer: torch.Tensor (capacity, emb_dim) on GPU
    - self.timestamps: np.ndarray (capacity,) on CPU for fast lookup
    - self.frame_hashes: Optional[np.ndarray] for validation
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
            capacity: Maximum number of embeddings to cache (e.g., 256)
            emb_dim: Embedding dimension (512 for radar, 2048 for scene)
            device: GPU device
            enable_validation: Whether to hash frames for cache validation
        """
        self.capacity = capacity
        self.emb_dim = emb_dim
        self.device = device
        self.enable_validation = enable_validation

        # GPU storage for embeddings (preallocated)
        self.buffer = torch.zeros(
            capacity, emb_dim,
            dtype=torch.float32,
            device=device
        )

        # CPU storage for metadata
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self.head = 0  # Current write position
        self.count = 0  # Total frames added

        # Optional: Frame validation hashes
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

        This is the CORE method that replaces:
          OLD: encode_all_32_frames()
          NEW: encode_1_frame() + retrieve_cached_sequence()

        Args:
            new_frame: (1, C, H, W) new frame to encode
            encoder: Encoder function (radar_encoder or yolo.embeds)
            seq_length: Length of sequence to return (e.g., 32)
            timestamp: Optional timestamp for this frame

        Returns:
            embeddings: (seq_length, emb_dim) sequence on GPU

        Example:
            # OLD CODE (slow):
            radar_embeds = radar_encoder(radar_tensor)  # (1, 32, 512)

            # NEW CODE (fast):
            latest_frame = radar_tensor[:, -1:]  # (1, 1, 3, 224, 224)
            radar_embeds = cache.add_and_get_sequence(
                latest_frame.squeeze(1),  # (1, 3, 224, 224)
                radar_encoder,
                seq_length=32
            )  # (32, 512) in 6ms instead of 175ms!
        """
        # 1. Encode the new frame
        with torch.no_grad():
            new_emb = encoder(new_frame)  # (1, emb_dim) or (1, 1, emb_dim)

            # Handle different encoder output formats
            if new_emb.dim() == 3:
                new_emb = new_emb.squeeze(1)  # (1, emb_dim)

            new_emb = new_emb.squeeze(0)  # (emb_dim,)

        # 2. Store in ring buffer
        self.buffer[self.head] = new_emb

        if timestamp is not None:
            self.timestamps[self.head] = timestamp

        # Optional: Hash validation
        if self.frame_hashes is not None:
            frame_hash = self._hash_frame(new_frame)
            self.frame_hashes[self.head] = frame_hash

        # 3. Update pointers
        self.head = (self.head + 1) % self.capacity
        self.count += 1

        # 4. Retrieve sequence of last seq_length embeddings
        if self.count < seq_length:
            # Not enough frames yet - pad with zeros
            valid = min(self.count, seq_length)
            padding = seq_length - valid

            if padding > 0:
                # Get all valid frames
                if self.head >= valid:
                    indices = torch.arange(
                        self.head - valid, self.head,
                        device=self.device
                    )
                else:
                    # Wrap around
                    indices = torch.cat([
                        torch.arange(
                            self.capacity - (valid - self.head),
                            self.capacity,
                            device=self.device
                        ),
                        torch.arange(0, self.head, device=self.device)
                    ])

                valid_embeds = self.buffer[indices]

                # Pad at beginning with zeros
                padded = torch.cat([
                    torch.zeros(
                        padding, self.emb_dim,
                        device=self.device,
                        dtype=torch.float32
                    ),
                    valid_embeds
                ], dim=0)

                return padded  # (seq_length, emb_dim)
            else:
                # Exactly enough frames
                indices = torch.arange(
                    self.head - valid, self.head,
                    device=self.device
                )
                return self.buffer[indices]

        # Normal case: retrieve last seq_length embeddings
        if self.head >= seq_length:
            # No wrap-around needed
            indices = torch.arange(
                self.head - seq_length,
                self.head,
                device=self.device
            )
        else:
            # Wrap around ring buffer
            indices = torch.cat([
                torch.arange(
                    self.capacity + self.head - seq_length,
                    self.capacity,
                    device=self.device
                ),
                torch.arange(0, self.head, device=self.device)
            ])

        return self.buffer[indices]  # (seq_length, emb_dim) - ALL ON GPU!

    def invalidate(self):
        """Clear cache (e.g., on round restart or game state change)."""
        self.head = 0
        self.count = 0
        self.buffer.zero_()
        self.timestamps.fill(0)

        if self.frame_hashes is not None:
            self.frame_hashes.fill(0)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "capacity": self.capacity,
            "count": self.count,
            "head": self.head,
            "filled_ratio": min(self.count / self.capacity, 1.0),
            "memory_mb": (self.capacity * self.emb_dim * 4) / (1024 * 1024),
        }

    def _hash_frame(self, frame: torch.Tensor) -> np.uint64:
        """
        Hash frame for validation (optional).

        Simple hash using mean and std to detect frame changes.
        """
        mean = frame.mean().item()
        std = frame.std().item()
        # Combine into uint64 hash
        hash_val = int((mean * 1e6 + std * 1e6)) % (2**64)
        return np.uint64(hash_val)


class AudioEmbeddingCache:
    """
    Cache for audio embeddings to avoid re-encoding.

    Unlike radar/scene which have per-frame caching, audio uses
    a different strategy:
    - Audio encoder processes full 30-sec window → 60 embeddings
    - Each embedding represents 0.5 sec (8000 samples @ 16kHz)
    - Re-encode only when 0.5 sec of new audio data accumulated
    """

    def __init__(
        self,
        embedding_duration_ms: int = 500,
        sample_rate: int = 16000,
        device: str = 'cuda'
    ):
        """
        Args:
            embedding_duration_ms: Duration per embedding (default 500ms)
            sample_rate: Audio sample rate (default 16kHz)
            device: GPU device
        """
        self.cached_embeddings: Optional[torch.Tensor] = None  # (1, 60, 512)
        self.last_encoded_position = 0
        self.embedding_step_samples = int(embedding_duration_ms / 1000 * sample_rate)  # 8000 samples
        self.device = device

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def get_embeddings(
        self,
        audio_tensor: torch.Tensor,
        current_position: int,
        encoder: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Get audio embeddings, using cache if possible.

        Args:
            audio_tensor: (1, 480000) audio samples (30 sec @ 16kHz)
            current_position: Current sample position in audio stream
            encoder: AudioEncoder model

        Returns:
            embeddings: (1, 60, 512) audio embeddings
        """
        samples_since_encode = current_position - self.last_encoded_position

        # Cache hit: less than 0.5 sec of new data
        if samples_since_encode < self.embedding_step_samples and self.cached_embeddings is not None:
            self.cache_hits += 1
            return self.cached_embeddings

        # Cache miss - re-encode full buffer
        self.cache_misses += 1

        with torch.no_grad():
            embeddings = encoder(audio_tensor)  # (1, 60, 512)

        self.cached_embeddings = embeddings
        self.last_encoded_position = current_position

        return embeddings

    def invalidate(self):
        """Clear cache."""
        self.cached_embeddings = None
        self.last_encoded_position = 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "memory_mb": 0.12 if self.cached_embeddings is not None else 0.0,  # 60 * 512 * 4 bytes
        }
