"""
TemporalCrossTransformer with Audio Domain Support.

Updated architecture to include audio embeddings from QuartzNet-style encoder.
Audio provides spatial awareness through game sounds (footsteps, gunshots, etc.)

New input: audio_seq (B, 60, 256) - 30 sec window with 0.5 sec step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class FlowActionHead(nn.Module):
    """
    Flow Matching Action Head for mouse delta prediction.

    Instead of direct regression (MSE), learns a velocity field that
    transforms noise into the target action. Handles multimodal action
    distributions (e.g., player can turn left OR right).

    Args:
        context_dim: dimension of context embedding from transformer
        action_dim: dimension of action output (2 for yaw/pitch)
        hidden_dim: hidden layer size
        noise_scale: scale of initial noise (smaller = less aggressive flow)
        num_steps: Euler integration steps during inference
    """

    def __init__(
        self,
        context_dim: int = 512,
        action_dim: int = 2,
        hidden_dim: int = 256,
        noise_scale: float = 0.3,
        num_steps: int = 5
    ):
        super().__init__()
        self.noise_scale = noise_scale
        self.action_dim = action_dim
        self.num_steps = num_steps

        # Time embedding via sinusoidal + MLP
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Main velocity prediction network
        input_dim = context_dim + action_dim + hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, context: torch.Tensor, noised_action: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity field v_θ(a_τ, τ, context).

        Args:
            context: (B, context_dim) - conditioning from transformer
            noised_action: (B, action_dim) - interpolated noisy action
            time_step: (B, 1) - τ ∈ [0, 1]

        Returns:
            velocity: (B, action_dim)
        """
        t_emb = self.time_embed(time_step)
        x = torch.cat([context, noised_action, t_emb], dim=-1)
        return self.net(x)

    def compute_loss(self, context: torch.Tensor, gt_action: torch.Tensor) -> torch.Tensor:
        """
        Flow matching training loss.

        Samples random τ and noise, computes MSE between predicted
        and target velocity vectors.

        Args:
            context: (B, context_dim)
            gt_action: (B, action_dim)

        Returns:
            loss: scalar
        """
        B = context.shape[0]
        device = context.device

        tau = torch.rand(B, 1, device=device)
        eps = torch.randn(B, self.action_dim, device=device) * self.noise_scale

        # Interpolate between noise and target
        a_tau = (1 - tau) * eps + tau * gt_action

        # Target velocity: direction from noise to target
        v_target = gt_action - eps

        # Predicted velocity
        v_pred = self.forward(context, a_tau, tau)

        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """
        Generate action via Euler integration of learned velocity field.

        Args:
            context: (B, context_dim)
            num_steps: override default integration steps

        Returns:
            action: (B, action_dim)
        """
        K = num_steps or self.num_steps
        B = context.shape[0]
        device = context.device

        # Start from scaled noise
        a = torch.randn(B, self.action_dim, device=device) * self.noise_scale

        # Euler integration
        for k in range(K):
            tau = torch.full((B, 1), k / K, device=device)
            v = self.forward(context, a, tau)
            a = a + v / K

        return a


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
        - mouse_embed: (B, d_model) - context for FlowActionHead
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
        # NOTE: policy_mouse_head removed — use FlowActionHead externally
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
            mouse_embed: (B, d_model) - context for FlowActionHead
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
        # mouse_embed is passed to FlowActionHead externally
        policy_keys = self.policy_keys_head(policy_keys_embed).view(B, 20)
        value = self.value_head(value_embed)

        return policy_mouse_embed, policy_keys, value


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
    flow_head = FlowActionHead(context_dim=512, noise_scale=0.3, num_steps=5).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flow_params = sum(p.numel() for p in flow_head.parameters() if p.requires_grad)
    print(f"Transformer parameters: {total_params:,}")
    print(f"FlowActionHead parameters: {flow_params:,}")

    # Create test inputs
    B = 2
    radar_seq = torch.randn(B, 129, 512).to(device)
    scene_seq = torch.randn(B, 16, 2048).to(device)
    audio_seq = torch.randn(B, 60, 256).to(device)
    detection_seq = torch.randn(B, 1, 100).to(device)
    action_seq = torch.randn(B, 16, 22).to(device)
    state_vec = torch.randn(B, 95).to(device)
    gt_mouse = torch.randn(B, 2).to(device)

    mouse_embed, policy_keys, value = model(
        radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq
    )

    print(f"mouse_embed: {mouse_embed.shape}")    # (B, 512)
    print(f"policy_keys: {policy_keys.shape}")     # (B, 20)
    print(f"value: {value.shape}")                 # (B, 1)

    # Test flow matching training loss
    loss = flow_head.compute_loss(mouse_embed, gt_mouse)
    print(f"Flow matching loss: {loss.item():.4f}")

    # Test flow matching inference
    sampled_action = flow_head.sample(mouse_embed)
    print(f"Sampled action: {sampled_action.shape}")  # (B, 2)

    # Test without audio (backward compatible)
    print("\n=== Testing TemporalCrossTransformer WITHOUT audio ===")
    model_no_audio = TemporalCrossTransformerNoAudio().to(device)

    mouse_embed, policy_keys, value = model_no_audio(
        radar_seq, scene_seq, detection_seq, action_seq, state_vec
    )

    print(f"mouse_embed: {mouse_embed.shape}")
    print(f"policy_keys: {policy_keys.shape}")
    print(f"value: {value.shape}")

    print("\nAll tests passed!")
