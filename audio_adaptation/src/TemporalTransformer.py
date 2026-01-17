"""
TemporalCrossTransformer with Audio Domain Support.

Updated architecture to include audio embeddings from QuartzNet-style encoder.
Audio provides spatial awareness through game sounds (footsteps, gunshots, etc.)

New input: audio_seq (B, 60, 256) - 30 sec window with 0.5 sec step
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def sinusoidal_position_encoding(seq_len, dim, device):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_out, _ = self.attn(query, key, value)
        x = self.norm1(query + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TemporalCrossTransformer(nn.Module):
    """
    Temporal Cross-Attention Transformer for CS2 AI Agent.

    Inputs:
        - radar_seq: (B, T_radar, 512) - radar embeddings
        - scene_seq: (B, T_scene, 2048) - scene embeddings from YOLO
        - audio_seq: (B, 60, 256) - audio embeddings (NEW)
        - detection_seq: (B, 1, 100) - current detections
        - action_seq: (B, 16, 22) - action history
        - state_vec: (B, 95) - game state

    Outputs:
        - policy_mouse: (B, 2) - yaw/pitch delta
        - policy_keys: (B, 20) - key probabilities
        - value: (B, 1) - value estimate
    """

    def __init__(
        self,
        # Existing dimensions
        radar_dim: int = 512,
        radar_seq: int = 129,
        scene_dim: int = 2048,
        scene_seq: int = 16,
        detection_dim: int = 100,
        detection_seq: int = 1,
        actions_dim: int = 22,
        actions_seq: int = 16,
        state_dim: int = 95,
        # Audio dimensions (NEW)
        audio_dim: int = 512,
        audio_seq: int = 60,
        use_audio: bool = True,
        # Transformer config
        d_model: int = 512,
        num_heads: int = 8,
        depth: int = 6,
        audio_depth: int = 2,  # Lighter encoder for audio
        ff_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.use_audio = use_audio
        self.d_model = d_model

        # === Projections ===
        self.radar_proj = nn.Linear(radar_dim, d_model)
        self.scene_proj = nn.Linear(scene_dim, d_model)
        self.detection_proj = nn.Linear(detection_dim, d_model)
        self.action_proj = nn.Linear(actions_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # Audio projection (NEW)
        if use_audio:
            self.audio_proj = nn.Linear(audio_dim, d_model)

        # === Positional encodings (sinusoidal) ===
        self.register_buffer("pos_embed_radar", sinusoidal_position_encoding(radar_seq, d_model, torch.device("cpu")))
        self.register_buffer("pos_embed_scene", sinusoidal_position_encoding(scene_seq, d_model, torch.device("cpu")))
        self.register_buffer("pos_embed_action", sinusoidal_position_encoding(actions_seq, d_model, torch.device("cpu")))

        # Audio positional encoding (NEW)
        if use_audio:
            self.register_buffer("pos_embed_audio", sinusoidal_position_encoding(audio_seq, d_model, torch.device("cpu")))

        # === Register tokens for different heads ===
        self.register_tokens_policy_mouse = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_tokens_policy_keys = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_tokens_value = nn.Parameter(torch.randn(1, 1, d_model))

        # === Transformer encoders ===
        self.radar_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout),
            num_layers=depth
        )
        self.scene_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout),
            num_layers=depth
        )

        # Audio encoder - lighter (NEW)
        if use_audio:
            self.audio_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout),
                num_layers=audio_depth
            )

        # === Cross attention ===
        self.cross_attn = CrossAttentionBlock(d_model, num_heads=num_heads, ff_mult=ff_mult, dropout=dropout)

        # === Output heads ===
        self.policy_mouse_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 2)  # yaw, pitch delta
        )
        self.policy_keys_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 20)  # 20 action classes
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(
        self,
        radar_seq: torch.Tensor,
        scene_seq: torch.Tensor,
        detection_seq: torch.Tensor,
        action_seq: torch.Tensor,
        state_vec: torch.Tensor,
        audio_seq: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional audio input.

        Args:
            radar_seq: (B, T_radar, 512)
            scene_seq: (B, T_scene, 2048)
            detection_seq: (B, 1, 100)
            action_seq: (B, 16, 22)
            state_vec: (B, 95)
            audio_seq: (B, 60, 256) or None

        Returns:
            policy_mouse: (B, 2)
            policy_keys: (B, 20)
            value: (B, 1)
        """
        B = radar_seq.shape[0]
        device = radar_seq.device

        # === Project and add positional encoding ===
        radar = self.radar_proj(radar_seq) + self.pos_embed_radar[:, :radar_seq.shape[1], :].to(device)
        scene = self.scene_proj(scene_seq) + self.pos_embed_scene[:, :scene_seq.shape[1], :].to(device)
        actions = self.action_proj(action_seq) + self.pos_embed_action[:, :action_seq.shape[1], :].to(device)
        state = self.state_proj(state_vec).unsqueeze(1)
        detections = self.detection_proj(detection_seq)

        # === Encode sequences ===
        radar = self.radar_encoder(radar)
        scene = self.scene_encoder(scene)

        # === Process audio if available (NEW) ===
        if self.use_audio and audio_seq is not None:
            audio = self.audio_proj(audio_seq) + self.pos_embed_audio[:, :audio_seq.shape[1], :].to(device)
            audio = self.audio_encoder(audio)
        else:
            audio = None

        # === Expand register tokens for batch ===
        register_tokens = torch.cat([
            self.register_tokens_policy_mouse,
            self.register_tokens_policy_keys,
            self.register_tokens_value
        ], dim=1).expand(B, -1, -1)  # (B, 3, d_model)

        # === Prepare context ===
        # Order: radar, scene, audio (if present), detections, actions, state
        if audio is not None:
            context = torch.cat([radar, scene, audio, detections, actions, state], dim=1)
        else:
            context = torch.cat([radar, scene, detections, actions, state], dim=1)

        # === Cross-attention: queries = register_tokens, key/value = context ===
        fused_tokens = self.cross_attn(register_tokens, context, context)  # (B, 3, d_model)

        # === Extract each token ===
        policy_mouse_embed = fused_tokens[:, 0, :]
        policy_keys_embed = fused_tokens[:, 1, :]
        value_embed = fused_tokens[:, 2, :]

        # === Forward through heads ===
        policy_mouse = self.policy_mouse_head(policy_mouse_embed).view(B, 2)
        policy_keys = self.policy_keys_head(policy_keys_embed).view(B, 20)
        value = self.value_head(value_embed)

        return policy_mouse, policy_keys, value


class TemporalCrossTransformerNoAudio(TemporalCrossTransformer):
    """
    Backward-compatible version without audio support.
    Identical to original TemporalCrossTransformer.
    """

    def __init__(self, **kwargs):
        kwargs['use_audio'] = False
        super().__init__(**kwargs)

    def forward(self, radar_seq, scene_seq, detection_seq, action_seq, state_vec):
        return super().forward(radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq=None)


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test with audio
    print("\n=== Testing TemporalCrossTransformer WITH audio ===")
    model = TemporalCrossTransformer(use_audio=True).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Create test inputs
    B = 2
    radar_seq = torch.randn(B, 129, 512).to(device)
    scene_seq = torch.randn(B, 16, 2048).to(device)
    audio_seq = torch.randn(B, 60, 256).to(device)  # NEW
    detection_seq = torch.randn(B, 1, 100).to(device)
    action_seq = torch.randn(B, 16, 22).to(device)
    state_vec = torch.randn(B, 95).to(device)

    policy_mouse, policy_keys, value = model(
        radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq
    )

    print(f"policy_mouse: {policy_mouse.shape}")  # (B, 2)
    print(f"policy_keys: {policy_keys.shape}")    # (B, 20)
    print(f"value: {value.shape}")                # (B, 1)

    # Test without audio (backward compatible)
    print("\n=== Testing TemporalCrossTransformer WITHOUT audio ===")
    model_no_audio = TemporalCrossTransformerNoAudio().to(device)

    policy_mouse, policy_keys, value = model_no_audio(
        radar_seq, scene_seq, detection_seq, action_seq, state_vec
    )

    print(f"policy_mouse: {policy_mouse.shape}")
    print(f"policy_keys: {policy_keys.shape}")
    print(f"value: {value.shape}")

    print("\nAll tests passed!")
