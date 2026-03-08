"""
StereoAudioEncoder — MobileNetV3-Large (mn10_as compatible) for stereo CS2 audio.

Architecture:
  StereoMelFrontend: (B, 2, T) → (B, 2, 128, T_frames)   [16kHz, hop=160, 128 mels]
  MobileNetV3-Large backbone (2-ch input, no global pool) → (B, 960, 4, 50) for 16sec
  Freq pool (mean over freq dim=4):                        → (B, 960, 50)
  Temporal pool 50 → target_embeddings=32:                 → (B, 32, 960)
  Projection Linear(960 → embed_dim) + LayerNorm:          → (B, 32, 512)

Input:  (B, 2, T)  stereo float32 waveform @ 16kHz, T=256,000 (16 sec)
Output: (B, 32, 512) — 32 temporal embeddings × 0.5 sec each

Pretrained weights (optional):
  load_pretrained_mn10as(path) — loads mn10_as.pt from EfficientAT repo
  https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as.pt
  First conv adapted: pretrained (16, 1, 3, 3) → stereo (16, 2, 3, 3) by w.repeat/2
  All other layers (15 InvertedResidual blocks + final conv) load as-is.

In Train.py: linspace(32 → 16) to match unified SEQ_LEN=16.

MobileNetV3-Large = mn10_as because:
  width_mult=1.0, same inverted_residual_setting, same SE squeeze factor
  torchvision implementation is the base EfficientAT adapted from.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TA
from torchvision.models import mobilenet_v3_large
from typing import Optional


class StereoMelFrontend(nn.Module):
    """
    Stereo mel-spectrogram frontend.

    (B, 2, T) → (B, 2, n_mels, T_frames)

    Processes L and R channels independently through the same MelSpectrogram,
    then normalizes per (channel, sample). Cross-channel ILD/HRTF patterns
    are preserved in the 2-channel output for the 2D CNN backbone.

    16kHz config:
      hop_length=160  → 10ms resolution, T=16sec → T_frames=1600
      n_fft=400       → 25ms window
      n_mels=128      → matches EfficientAT default
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 400,
                 hop_length: int = 160, n_mels: int = 128):
        super().__init__()
        self.mel_spec = TA.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=0.0, f_max=8000.0, power=2.0,
        )
        self.amplitude_to_db = TA.AmplitudeToDB(stype='power', top_db=80)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, T) stereo waveform
        Returns:
            mel: (B, 2, n_mels, T_frames) normalized per channel per sample
        """
        B, C, T = x.shape
        x_flat = x.reshape(B * C, T)                              # (B*2, T)
        mel = self.amplitude_to_db(self.mel_spec(x_flat))         # (B*2, n_mels, T')
        mel = mel.reshape(B, C, mel.shape[1], mel.shape[2])       # (B, 2, n_mels, T')

        # Per-channel normalization — preserves ILD between L and R
        mean = mel.mean(dim=(2, 3), keepdim=True)
        std = mel.std(dim=(2, 3), keepdim=True) + 1e-6
        return (mel - mean) / std


class StereoAudioEncoder(nn.Module):
    """
    Stereo audio encoder: MobileNetV3-Large backbone (mn10_as architecture)
    with 2-channel stereo mel input and temporal embedding output.

    Designed for CS2 spatial audio: footsteps, gunshots, bomb sounds.
    The 2D CNN processes (B, 2, 128, T_frames) stereo spectrograms,
    learning ILD (level difference) and HRTF patterns across L/R channels.

    Input:  (B, 2, T) stereo waveform, T=256,000 (16 sec @ 16kHz)
    Output: (B, 32, 512)

    Feature map pipeline (16sec input):
      (B, 2, 128, 1600) → backbone → (B, 960, 4, 50)
      freq pool:  (B, 960, 50)
      time pool:  (B, 960, 32)
      permute:    (B, 32, 960)
      project:    (B, 32, 512)

    Params: ~4.88M (backbone 4.57M + frontend 0M + projection 0.31M)
    MACs:   ~0.9B for 16sec stereo input
    """

    AUDIO_DURATION_SEC = 16.0
    SAMPLE_RATE = 16000
    EXPECTED_SAMPLES = int(AUDIO_DURATION_SEC * SAMPLE_RATE)   # 256,000
    EXPECTED_STEREO_SHAPE = (2, EXPECTED_SAMPLES)

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        embed_dim: int = 512,
        target_embeddings: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.target_embeddings = target_embeddings

        # Stereo mel frontend
        self.frontend = StereoMelFrontend(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=n_mels,
        )

        # MobileNetV3-Large backbone (same architecture as mn10_as, width_mult=1.0)
        # features[0..16]: 17 layers, outputs (B, 960, h, w) before global pool
        _mv3 = mobilenet_v3_large(weights=None)

        # Adapt first conv: 3 channels → 2 channels (stereo)
        first_conv = _mv3.features[0][0]   # Conv2d(3, 16, ...)
        new_first_conv = nn.Conv2d(
            in_channels=2,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        _mv3.features[0][0] = new_first_conv

        # Use only features (no avgpool, no classifier)
        # features[-1] outputs (B, 960, 4, 50) for (B, 2, 128, 1600) input
        self.backbone = _mv3.features   # nn.Sequential, len=17

        # Project: 960 → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(960, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, 2, T) stereo float32 waveform, T=256,000

        Returns:
            embeddings: (B, target_embeddings, embed_dim)  e.g. (B, 32, 512)
        """
        # Stereo mel: (B, 2, 128, ~1600)
        mel = self.frontend(waveform)

        # Backbone (no global pool): (B, 960, 4, ~50)
        feat = self.backbone(mel)

        # Pool over frequency dim (height=4 after 5× stride-2): (B, 960, ~50)
        feat = feat.mean(dim=2)

        # Temporal pool → fixed target_embeddings: (B, 960, 32)
        feat = F.adaptive_avg_pool1d(feat, self.target_embeddings)

        # (B, 32, 960)
        feat = feat.permute(0, 2, 1)

        return self.projection(feat)  # (B, 32, 512)

    def load_pretrained_mn10as(self, path: Optional[str] = None) -> None:
        """
        Load pretrained mn10_as weights from EfficientAT.

        Downloads automatically if path is None:
          https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as.pt

        First conv adaptation:
          pretrained: features.0.0.weight shape (16, 1, 3, 3)  [mono AudioSet]
          new:        features.0.0.weight shape (16, 2, 3, 3)  [stereo]
          method:     new_w = w.repeat(1, 2, 1, 1) / 2.0
          both channels get identical AudioSet-pretrained filters, averaged

        All other 16 backbone layers load without modification (keys are identical
        between torchvision MobileNetV3-Large and EfficientAT mn10_as).
        """
        url = ('https://github.com/fschmid56/EfficientAT/releases/'
               'download/v0.0.1/mn10_as.pt')

        if path is None:
            print(f'[StereoAudioEncoder] Downloading mn10_as pretrained from GitHub...')
            ckpt = torch.hub.load_state_dict_from_url(
                url, map_location='cpu', progress=True,
            )
        else:
            ckpt = torch.load(path, map_location='cpu', weights_only=True)

        # Unwrap nested dicts
        for key in ('state_dict', 'model_state_dict', 'model'):
            if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

        # Adapt first conv: (16, 1, 3, 3) → (16, 2, 3, 3)
        first_key = 'features.0.0.weight'
        if first_key in ckpt:
            w = ckpt[first_key]                         # (16, 1, 3, 3)
            ckpt[first_key] = w.repeat(1, 2, 1, 1) / 2.0   # (16, 2, 3, 3)
            print(f'[StereoAudioEncoder] First conv adapted: '
                  f'{list(w.shape)} → {list(ckpt[first_key].shape)}')

        # Strip 'features.' prefix to match self.backbone state dict keys
        # (self.backbone = _mv3.features, so keys are '0.0.weight' etc.)
        backbone_state = {}
        for k, v in ckpt.items():
            if k.startswith('features.'):
                backbone_state[k[len('features.'):]] = v   # e.g. '0.0.weight'

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        n_loaded = len(backbone_state) - len(unexpected)
        print(f'[StereoAudioEncoder] Pretrained mn10_as loaded: '
              f'{n_loaded}/{len(backbone_state)} keys | '
              f'missing={len(missing)} unexpected={len(unexpected)}')
        if missing:
            print(f'  Missing (first 5): {missing[:5]}')
        if unexpected:
            print(f'  Unexpected (first 5): {unexpected[:5]}')


if __name__ == '__main__':
    encoder = StereoAudioEncoder(embed_dim=512, target_embeddings=32)
    total = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f'StereoAudioEncoder params: {total:,}')   # ~4.88M

    # 16 sec stereo @ 16kHz
    x = torch.randn(2, 2, 256_000)   # (B=2, channels=2, T)
    out = encoder(x)
    print(f'Output shape: {out.shape}')   # (2, 32, 512)

    # Test pretrained loading (requires internet or local path)
    # encoder.load_pretrained_mn10as()
