"""
AudioEncoder — Upgraded QuartzNet-style encoder for CS2 game audio.

Upgrade from 3-block (1.3M) to 4-block wider architecture (~3.5M):
  conv_init: 64 mel → 256 ch  (was 128)
  block1: QuartzNetBlock(256, 256, k=33, R=5)
  block2: QuartzNetBlock(256, 384, k=39, R=5)
  block3: QuartzNetBlock(384, 384, k=51, R=5)
  block4: QuartzNetBlock(384, 512, k=63, R=5)  ← NEW
  temporal_pool: target=32 embeddings (16 sec @ 0.5s step)
  projection: Linear(512, embed_dim=512)

Input:  16 sec mono audio @ 16kHz = 256,000 samples
Output: (B, 32, 512)  — 32 embeddings, each 0.5 sec

In Train.py: linspace(32 → 16) to match unified SEQ_LEN=16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Optional


class TCSConv1d(nn.Module):
    """Time-Channel Separable Convolution 1D (depthwise + pointwise)."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, stride=stride,
                                   padding=padding, dilation=dilation,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class QuartzNetBlock(nn.Module):
    """QuartzNet block: R × [TCS-Conv → BN → ReLU] + residual."""
    def __init__(self, in_ch, out_ch, kernel_size, repeat=5, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2

        layers = []
        for i in range(repeat):
            ch_in = in_ch if i == 0 else out_ch
            layers.append(TCSConv1d(ch_in, out_ch, kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(out_ch))
            if i < repeat - 1:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.main = nn.Sequential(*layers)

        if in_ch != out_ch:
            self.res_conv = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.res_conv = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main(x) + self.res_conv(x)
        return self.activation(out)


class MelSpectrogramFrontend(nn.Module):
    """Mel-spectrogram frontend: waveform → (B, n_mels, T_frames)."""
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 n_mels=64, f_min=0.0, f_max=8000.0, normalize=True):
        super().__init__()
        self.hop_length = hop_length
        self.normalize = normalize
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max, power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)

    def forward(self, waveform):
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        mel = self.amplitude_to_db(self.mel_spec(waveform))
        if self.normalize:
            mean = mel.mean(dim=(1, 2), keepdim=True)
            std = mel.std(dim=(1, 2), keepdim=True) + 1e-6
            mel = (mel - mean) / std
        return mel  # (B, n_mels, T_frames)


class TemporalPooling(nn.Module):
    """Pool temporal dim to fixed number of embeddings via avg pooling."""
    def __init__(self, target_embeddings=32):
        super().__init__()
        self.target_embeddings = target_embeddings

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        pool_size = T // self.target_embeddings
        T_trimmed = pool_size * self.target_embeddings
        x = x[:, :, :T_trimmed]
        x = x.view(B, C, self.target_embeddings, pool_size)
        pooled = x.mean(dim=-1)           # (B, C, target_embeddings)
        return pooled.permute(0, 2, 1)    # (B, target_embeddings, C)


class AudioEncoder(nn.Module):
    """
    Upgraded QuartzNet-style Audio Encoder.

    Input:  16 sec mono audio @ 16kHz  (256,000 samples)
    Output: (B, 32, embed_dim)          32 embeddings × 0.5 sec each

    Architecture: ~3.5M parameters
        MelSpec frontend (64 bands)
        conv_init: 64 → 256 ch
        block1:    256 → 256, k=33, R=5
        block2:    256 → 384, k=39, R=5
        block3:    384 → 384, k=51, R=5
        block4:    384 → 512, k=63, R=5
        temporal_pool: → 32 embeddings
        projection: 512 → embed_dim
    """

    AUDIO_DURATION_SEC = 16.0    # 16 sec audio window
    SAMPLE_RATE = 16000
    EXPECTED_SAMPLES = int(AUDIO_DURATION_SEC * SAMPLE_RATE)  # 256,000

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        embed_dim: int = 512,
        target_embeddings: int = 32,   # 16 sec / 0.5 sec = 32
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.target_embeddings = target_embeddings

        # Mel frontend
        self.frontend = MelSpectrogramFrontend(
            sample_rate=sample_rate, n_fft=512, hop_length=160,
            n_mels=n_mels, f_min=0.0, f_max=8000.0
        )

        # Initial conv: n_mels → 256
        self.conv_init = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=33, stride=2, padding=16, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # QuartzNet blocks (4 blocks, wider channels)
        self.block1 = QuartzNetBlock(256, 256, kernel_size=33, repeat=5, dropout=dropout)
        self.block2 = QuartzNetBlock(256, 384, kernel_size=39, repeat=5, dropout=dropout)
        self.block3 = QuartzNetBlock(384, 384, kernel_size=51, repeat=5, dropout=dropout)
        self.block4 = QuartzNetBlock(384, 512, kernel_size=63, repeat=5, dropout=dropout)  # NEW

        # Temporal pooling → target_embeddings
        self.temporal_pool = TemporalPooling(target_embeddings=target_embeddings)

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) or (B, 1, T)  T = 256,000 for 16 sec @ 16kHz

        Returns:
            embeddings: (B, target_embeddings, embed_dim)  e.g. (B, 32, 512)
        """
        mel = self.frontend(waveform)     # (B, 64, ~1600)
        x = self.conv_init(mel)           # (B, 256, ~800)
        x = self.block1(x)                # (B, 256, ~800)
        x = self.block2(x)                # (B, 384, ~800)
        x = self.block3(x)                # (B, 384, ~800)
        x = self.block4(x)                # (B, 512, ~800)
        pooled = self.temporal_pool(x)    # (B, 32, 512)
        return self.projection(pooled)    # (B, 32, embed_dim)


if __name__ == "__main__":
    encoder = AudioEncoder(embed_dim=512, target_embeddings=32)
    total = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"AudioEncoder params: {total:,}")  # ~3.5M

    # 16 sec @ 16kHz
    x = torch.randn(2, 256000)
    out = encoder(x)
    print(f"Output shape: {out.shape}")   # (2, 32, 512)
