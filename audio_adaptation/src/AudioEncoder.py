"""
AudioEncoder - QuartzNet-style encoder for CS2 game audio.

Architecture based on QuartzNet (NVIDIA) with modifications for:
- Game audio (not speech)
- 30 second window
- 0.5 second embedding step (60 embeddings per window)
- Real-time friendly processing

Key specs:
- Input: 30 sec mono audio @ 16kHz = 480,000 samples
- Output: (B, 60, 256) embeddings
- Each embedding represents 0.5 sec of audio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple


class TCSConv1d(nn.Module):
    """
    Time-Channel Separable Convolution (1D).
    Depthwise separable convolution for efficiency.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        # Pointwise convolution
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class QuartzNetBlock(nn.Module):
    """
    QuartzNet Block with residual connections.

    Structure: R x [TCS-Conv → BN → ReLU] + Residual
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        repeat: int = 5,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        residual: bool = True
    ):
        super().__init__()
        self.residual = residual

        # Padding for same output size (when stride=1)
        padding = (kernel_size - 1) // 2 * dilation

        # Main path: R repetitions
        layers = []
        for i in range(repeat):
            ch_in = in_channels if i == 0 else out_channels

            layers.append(TCSConv1d(
                ch_in, out_channels, kernel_size,
                stride=stride if i == 0 else 1,
                padding=padding,
                dilation=dilation
            ))
            layers.append(nn.BatchNorm1d(out_channels))

            # ReLU after all except last
            if i < repeat - 1:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.main = nn.Sequential(*layers)

        # Residual path
        if residual:
            if in_channels != out_channels or stride != 1:
                self.res_conv = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )
            else:
                self.res_conv = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)

        if self.residual:
            res = self.res_conv(x)
            out = out + res

        out = self.activation(out)
        return out


class MelSpectrogramFrontend(nn.Module):
    """
    Mel-spectrogram frontend for audio processing.
    Converts raw waveform to mel-spectrogram.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,  # 10ms at 16kHz
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        normalize: bool = True
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.normalize = normalize

        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0
        )

        # Log compression
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, 1, T) or (B, T) raw audio
        Returns:
            mel_spec: (B, n_mels, time_frames)
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # (B, T)

        # Compute mel-spectrogram
        mel = self.mel_spec(waveform)  # (B, n_mels, time_frames)

        # Convert to dB scale
        mel = self.amplitude_to_db(mel)

        # Normalize per-sample
        if self.normalize:
            # Mean-std normalization
            mean = mel.mean(dim=(1, 2), keepdim=True)
            std = mel.std(dim=(1, 2), keepdim=True) + 1e-6
            mel = (mel - mean) / std

        return mel


class TemporalPooling(nn.Module):
    """
    Pools temporal dimension to create fixed-step embeddings.

    For 30 sec @ 16kHz with hop_length=160:
    - Input frames: ~3000
    - Output embeddings: 60 (0.5 sec each)
    - Pool size: 50 frames per embedding
    """
    def __init__(
        self,
        target_embeddings: int = 60,
        pool_type: str = 'avg'  # 'avg', 'max', 'attention'
    ):
        super().__init__()
        self.target_embeddings = target_embeddings
        self.pool_type = pool_type

        if pool_type == 'attention':
            # Learnable attention pooling
            self.attention = nn.Sequential(
                nn.Linear(256, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) feature sequence
        Returns:
            pooled: (B, target_embeddings, C)
        """
        B, C, T = x.shape

        # Calculate pool size
        pool_size = T // self.target_embeddings

        # Trim to exact multiple
        T_trimmed = pool_size * self.target_embeddings
        x = x[:, :, :T_trimmed]

        # Reshape for pooling: (B, C, num_embeddings, pool_size)
        x = x.view(B, C, self.target_embeddings, pool_size)

        if self.pool_type == 'avg':
            pooled = x.mean(dim=-1)  # (B, C, num_embeddings)
        elif self.pool_type == 'max':
            pooled = x.max(dim=-1)[0]
        elif self.pool_type == 'attention':
            # (B, C, num_embeddings, pool_size) → (B, num_embeddings, pool_size, C)
            x_t = x.permute(0, 2, 3, 1)
            attn_weights = self.attention(x_t).squeeze(-1)  # (B, num_embeddings, pool_size)
            attn_weights = F.softmax(attn_weights, dim=-1)
            pooled = (x_t * attn_weights.unsqueeze(-1)).sum(dim=2)  # (B, num_embeddings, C)
            return pooled

        # (B, C, num_embeddings) → (B, num_embeddings, C)
        pooled = pooled.permute(0, 2, 1)

        return pooled


class AudioEncoder(nn.Module):
    """
    QuartzNet-style Audio Encoder for CS2 game audio.

    Input: 30 seconds of mono audio @ 16kHz
    Output: 60 embeddings of dimension 256 (0.5 sec each)

    Architecture:
    1. Mel-spectrogram frontend (64 mel bands)
    2. Initial Conv layer
    3. QuartzNet blocks (B1-B3)
    4. Temporal pooling to 60 embeddings
    5. Final projection
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        embed_dim: int = 512,
        target_embeddings: int = 60,
        dropout: float = 0.1,
        pool_type: str = 'avg'
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.embed_dim = embed_dim
        self.target_embeddings = target_embeddings

        # Audio duration for target_embeddings at 0.5 sec step
        self.audio_duration = target_embeddings * 0.5  # 30 seconds
        self.expected_samples = int(self.audio_duration * sample_rate)  # 480000

        # 1. Mel-spectrogram frontend
        self.frontend = MelSpectrogramFrontend(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=160,  # 10ms
            n_mels=n_mels,
            f_min=0.0,
            f_max=8000.0
        )

        # 2. Initial convolution (C1)
        self.conv_init = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=33, stride=2, padding=16, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # 3. QuartzNet blocks
        # B1: 128 → 128, kernel=33, 5 repeats
        self.block1 = QuartzNetBlock(128, 128, kernel_size=33, repeat=5, dropout=dropout)

        # B2: 128 → 256, kernel=39, 5 repeats
        self.block2 = QuartzNetBlock(128, 256, kernel_size=39, repeat=5, dropout=dropout)

        # B3: 256 → 256, kernel=51, 5 repeats
        self.block3 = QuartzNetBlock(256, 256, kernel_size=51, repeat=5, dropout=dropout)

        # 4. Temporal pooling
        self.temporal_pool = TemporalPooling(
            target_embeddings=target_embeddings,
            pool_type=pool_type
        )

        # 5. Final projection and normalization
        self.projection = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(
        self,
        waveform: torch.Tensor,
        return_all_frames: bool = False
    ) -> torch.Tensor:
        """
        Args:
            waveform: (B, 1, T) or (B, T) raw audio, T = 480000 for 30 sec
            return_all_frames: if True, return all frames before pooling

        Returns:
            embeddings: (B, 60, 256) audio embeddings
        """
        # Frontend: waveform → mel-spectrogram
        mel = self.frontend(waveform)  # (B, 64, ~3000)

        # Initial conv
        x = self.conv_init(mel)  # (B, 128, ~1500)

        # QuartzNet blocks
        x = self.block1(x)  # (B, 128, ~1500)
        x = self.block2(x)  # (B, 256, ~1500)
        x = self.block3(x)  # (B, 256, ~1500)

        if return_all_frames:
            # Return before pooling for streaming inference
            return x.permute(0, 2, 1)  # (B, T, 256)

        # Temporal pooling: (B, 256, T) → (B, 60, 256)
        pooled = self.temporal_pool(x)

        # Final projection
        embeddings = self.projection(pooled)  # (B, 60, 256)

        return embeddings

    def get_embedding_at_time(
        self,
        waveform: torch.Tensor,
        time_sec: float
    ) -> torch.Tensor:
        """
        Get embedding for a specific time point.

        Args:
            waveform: full audio waveform
            time_sec: time in seconds (0-30)

        Returns:
            embedding: (B, 256) single embedding
        """
        embeddings = self.forward(waveform)  # (B, 60, 256)

        # Convert time to embedding index
        idx = int(time_sec / 0.5)
        idx = min(max(idx, 0), self.target_embeddings - 1)

        return embeddings[:, idx, :]


class StreamingAudioEncoder(nn.Module):
    """
    Streaming-friendly wrapper for AudioEncoder.
    Maintains a rolling buffer of audio for real-time inference.

    Usage:
        encoder = StreamingAudioEncoder()

        # Feed audio chunks
        for chunk in audio_stream:
            embeddings = encoder.process_chunk(chunk)
            # embeddings contains new 0.5s embeddings since last call
    """
    def __init__(
        self,
        encoder: Optional[AudioEncoder] = None,
        buffer_duration: float = 30.0,
        sample_rate: int = 16000
    ):
        super().__init__()

        self.encoder = encoder or AudioEncoder(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(buffer_duration * sample_rate)

        # Rolling buffer
        self.register_buffer(
            'audio_buffer',
            torch.zeros(1, self.buffer_size)
        )
        self.buffer_position = 0
        self.last_embedding_position = 0

    def reset(self):
        """Reset the buffer."""
        self.audio_buffer.zero_()
        self.buffer_position = 0
        self.last_embedding_position = 0

    def process_chunk(self, chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process an audio chunk and return new embeddings if available.

        Args:
            chunk: (chunk_size,) or (1, chunk_size) audio samples

        Returns:
            new_embeddings: (num_new, 256) or None if no new embeddings
        """
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)

        chunk_size = chunk.shape[-1]

        # Add chunk to buffer (circular)
        if self.buffer_position + chunk_size <= self.buffer_size:
            self.audio_buffer[0, self.buffer_position:self.buffer_position + chunk_size] = chunk
            self.buffer_position += chunk_size
        else:
            # Shift buffer and add new chunk
            shift = chunk_size
            self.audio_buffer[0, :-shift] = self.audio_buffer[0, shift:].clone()
            self.audio_buffer[0, -chunk_size:] = chunk
            self.buffer_position = self.buffer_size

        # Check if we have new 0.5s window
        samples_per_embedding = int(0.5 * self.sample_rate)
        current_embedding_idx = self.buffer_position // samples_per_embedding
        last_embedding_idx = self.last_embedding_position // samples_per_embedding

        if current_embedding_idx > last_embedding_idx:
            # Get all embeddings
            embeddings = self.encoder(self.audio_buffer.unsqueeze(0))  # (1, 60, 256)

            # Return only new embeddings
            num_new = current_embedding_idx - last_embedding_idx
            new_embeddings = embeddings[0, -num_new:, :]  # (num_new, 256)

            self.last_embedding_position = self.buffer_position
            return new_embeddings

        return None


def test_audio_encoder():
    """Test AudioEncoder with synthetic audio."""
    print("Testing AudioEncoder...")

    # Create encoder
    encoder = AudioEncoder(
        sample_rate=16000,
        embed_dim=256,
        target_embeddings=60
    )

    # Synthetic audio: 30 seconds @ 16kHz
    batch_size = 2
    duration = 30.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)

    # Create test waveform (sine waves + noise)
    t = torch.linspace(0, duration, num_samples)
    waveform = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz sine
    waveform = waveform + 0.1 * torch.randn_like(waveform)  # Add noise
    waveform = waveform.unsqueeze(0).repeat(batch_size, 1)  # (B, T)

    print(f"Input shape: {waveform.shape}")

    # Forward pass
    with torch.no_grad():
        embeddings = encoder(waveform)

    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: ({batch_size}, 60, 256)")

    assert embeddings.shape == (batch_size, 60, 256), "Shape mismatch!"
    print("Test passed!")

    # Test streaming encoder
    print("\nTesting StreamingAudioEncoder...")
    streaming = StreamingAudioEncoder(encoder)

    chunk_size = 8000  # 0.5 sec
    for i in range(4):
        chunk = waveform[0, i*chunk_size:(i+1)*chunk_size]
        new_emb = streaming.process_chunk(chunk)
        if new_emb is not None:
            print(f"Chunk {i}: got {new_emb.shape[0]} new embedding(s)")

    print("Streaming test passed!")


if __name__ == "__main__":
    test_audio_encoder()
