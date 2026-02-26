"""
TemporalCrossTransformer — Unified Transformer with cross-attention compression.

Scene (64 tokens) and actions (64 tokens) are compressed to 16 via ModalityCompressor
(cross-attention with 16 learnable queries), then concatenated with other modalities
for the main unified Transformer encoder.

Input formats (raw → transformer):
  radar_seq:     (B, 16, 512)   ← 32 raw @1Hz → linspace 16         → 16 tokens
  scene_seq:     (B, 64, 512)   ← 64 raw @~16Hz → cross-attn 64→16  → 16 tokens
  audio_seq:     (B, 16, 512)   ← optional, 32 raw → linspace 16     → 16 tokens
  detection_seq: (B, 16, 100)   ← 64 raw → linspace 16               → 16 tokens
  action_seq:    (B, 64, 22)    ← 64 raw @~16Hz → cross-attn 64→16  → 16 tokens
  state_vec:     (B, 95)                                              →  1 token

Context after compression + concatenation:
  (B, 81, 384) with audio:    16+16+16+16+16+1 = 81
  (B, 65, 384) without audio: 16+16+16+16+1    = 65

FLOPs vs 145-token baseline: −37% net (950M saved − 173M overhead).
FlowActionHead context_dim MUST match d_model=384.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def sinusoidal_position_encoding(seq_len: int, dim: int, device) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq_len, dim)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
                                          batch_first=True)
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


class ModalityCompressor(nn.Module):
    """
    Cross-attention compression: T source tokens → N compressed tokens (N < T).

    N learnable query tokens attend to T source tokens (already projected + pos-encoded).
    Single cross-attention block + FFN. Learns which temporal positions to aggregate.

    Typical usage: 64 scene or action tokens → 16 compressed tokens.
    FLOPs: ~86M per module (N=16, T=64, d=384) vs 237M saved per Transformer layer.
    """

    def __init__(self, n_out: int, d_model: int, num_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_out = n_out
        # Learnable query tokens (no positional encoding — queries learn their own context)
        self.queries = nn.Parameter(torch.randn(1, n_out, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)  source tokens (projected + pos-encoded + modality-typed)
        Returns:
            (B, n_out, d_model)  compressed tokens
        """
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)          # (B, n_out, d_model)
        attn_out, _ = self.cross_attn(q, x, x)
        q = self.norm1(q + self.dropout(attn_out))
        q = self.norm2(q + self.dropout(self.ff(q)))
        return q                                     # (B, n_out, d_model)


class FlowActionHead(nn.Module):
    """
    Flow Matching Action Head for mouse delta prediction.

    context_dim MUST equal d_model of TemporalCrossTransformer (384).
    """

    def __init__(
        self,
        context_dim: int = 384,   # MUST match d_model
        action_dim: int = 2,
        hidden_dim: int = 256,
        noise_scale: float = 0.3,
        num_steps: int = 5
    ):
        super().__init__()
        self.noise_scale = noise_scale
        self.action_dim = action_dim
        self.num_steps = num_steps

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        input_dim = context_dim + action_dim + hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, context: torch.Tensor, noised_action: torch.Tensor,
                time_step: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(time_step)
        x = torch.cat([context, noised_action, t_emb], dim=-1)
        return self.net(x)

    def compute_loss(self, context: torch.Tensor, gt_action: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        device = context.device
        tau = torch.rand(B, 1, device=device)
        eps = torch.randn(B, self.action_dim, device=device) * self.noise_scale
        a_tau = (1 - tau) * eps + tau * gt_action
        v_target = gt_action - eps
        v_pred = self.forward(context, a_tau, tau)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        K = num_steps or self.num_steps
        B = context.shape[0]
        device = context.device
        a = torch.randn(B, self.action_dim, device=device) * self.noise_scale
        for k in range(K):
            tau = torch.full((B, 1), k / K, device=device)
            v = self.forward(context, a, tau)
            a = a + v / K
        return a


# Modality indices for nn.Embedding
MODALITY_RADAR   = 0
MODALITY_SCENE   = 1
MODALITY_AUDIO   = 2
MODALITY_DET     = 3
MODALITY_ACTION  = 4
MODALITY_STATE   = 5
NUM_MODALITIES   = 6


class TemporalCrossTransformer(nn.Module):
    """
    Temporal Cross-Attention Transformer for CS2 AI Agent.

    Scene (64 tokens) and actions (64 tokens) are compressed to 16 via ModalityCompressor
    before the main unified Transformer. All other modalities are already 16 tokens.

    Inputs:
        radar_seq:     (B, 16, radar_dim=512)   — 16 tokens (already linspaced)
        scene_seq:     (B, 64, scene_dim=512)   — 64 tokens → compressed to 16
        audio_seq:     (B, 16, audio_dim=512)   — optional, 16 tokens
        detection_seq: (B, 16, detection_dim=100) — 16 tokens (already linspaced)
        action_seq:    (B, 64, actions_dim=22)  — 64 tokens → compressed to 16
        state_vec:     (B, state_dim=95)        — 1 token

    After compression: 16+16+16+16+16+1 = 81 tokens (with audio), 65 without.

    Outputs:
        mouse_embed:  (B, d_model=384)  → pass to FlowActionHead
        policy_keys:  (B, 20)
        value:        (B, 1)
    """

    def __init__(
        self,
        # Input dimensions
        radar_dim: int = 512,
        scene_dim: int = 512,
        detection_dim: int = 100,
        actions_dim: int = 22,
        state_dim: int = 95,
        # Audio
        audio_dim: int = 512,
        use_audio: bool = True,
        # Sequence lengths per modality (input sizes before compression)
        seq_len: int = 16,             # radar, detection, audio: already 16
        scene_seq_len: int = 64,       # scene input: 64 tokens (compressed → seq_len)
        actions_seq_len: int = 64,     # action input: 64 tokens (compressed → seq_len)
        # Transformer config
        d_model: int = 384,
        num_heads: int = 8,
        depth: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_audio = use_audio
        self.d_model = d_model
        self.seq_len = seq_len
        self.scene_seq_len = scene_seq_len
        self.actions_seq_len = actions_seq_len

        # Input projections (all → d_model)
        self.radar_proj = nn.Linear(radar_dim, d_model)
        self.scene_proj = nn.Linear(scene_dim, d_model)
        self.detection_proj = nn.Linear(detection_dim, d_model)
        self.action_proj = nn.Linear(actions_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        if use_audio:
            self.audio_proj = nn.Linear(audio_dim, d_model)

        # Sinusoidal positional encodings (separate per input length)
        self.register_buffer(
            "pos_embed_seq",
            sinusoidal_position_encoding(seq_len, d_model, torch.device("cpu"))
        )  # (1, 16, 384)  — radar, detection, audio
        self.register_buffer(
            "pos_embed_scene",
            sinusoidal_position_encoding(scene_seq_len, d_model, torch.device("cpu"))
        )  # (1, 64, 384)  — scene before compression
        self.register_buffer(
            "pos_embed_actions",
            sinusoidal_position_encoding(actions_seq_len, d_model, torch.device("cpu"))
        )  # (1, 64, 384)  — actions before compression

        # Modality type embeddings: radar=0, scene=1, audio=2, det=3, action=4, state=5
        self.modality_embedding = nn.Embedding(NUM_MODALITIES, d_model)

        # Cross-attention compressors: 64 → 16 tokens each
        # Applied BEFORE unified_encoder; pos + modality info already baked in
        self.scene_compressor  = ModalityCompressor(seq_len, d_model, num_heads, ff_mult, dropout)
        self.action_compressor = ModalityCompressor(seq_len, d_model, num_heads, ff_mult, dropout)

        # Single unified Transformer encoder over all modality tokens (81 after compression)
        # dim_feedforward explicitly set to d_model*ff_mult (fixes PyTorch default-2048 bug)
        self.unified_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * ff_mult,  # 384*4=1536
                batch_first=True,
                dropout=dropout,
                norm_first=False
            ),
            num_layers=depth
        )

        # Register tokens for 3 output heads
        self.register_tokens_policy_mouse = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_tokens_policy_keys = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_tokens_value = nn.Parameter(torch.randn(1, 1, d_model))

        # Cross-attention: 3 register tokens attend to unified context
        self.cross_attn = CrossAttentionBlock(d_model, num_heads=num_heads,
                                              ff_mult=ff_mult, dropout=dropout)

        # Output heads
        self.policy_keys_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 20)
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
        Args:
            radar_seq:     (B, 16, 512)
            scene_seq:     (B, 64, 512)  — compressed 64→16 via ModalityCompressor
            detection_seq: (B, 16, 100)
            action_seq:    (B, 64, 22)   — compressed 64→16 via ModalityCompressor
            state_vec:     (B, 95)
            audio_seq:     (B, 16, 512) or None

        Returns:
            mouse_embed:  (B, 384)  — context for FlowActionHead
            policy_keys:  (B, 20)
            value:        (B, 1)
        """
        B = radar_seq.shape[0]
        device = radar_seq.device

        pos16     = self.pos_embed_seq.to(device)      # (1, 16, 384)
        pos64_sc  = self.pos_embed_scene.to(device)    # (1, 64, 384)
        pos64_act = self.pos_embed_actions.to(device)  # (1, 64, 384)
        mod = self.modality_embedding.weight.to(device)  # (6, 384)

        # Radar, detection, audio: no compression needed (already 16 tokens)
        radar = self.radar_proj(radar_seq) + pos16 + mod[MODALITY_RADAR]   # (B,16,384)
        det   = self.detection_proj(detection_seq) + pos16 + mod[MODALITY_DET]  # (B,16,384)
        state = self.state_proj(state_vec).unsqueeze(1) + mod[MODALITY_STATE]   # (B, 1,384)

        # Scene: project + pos64 + modality → cross-attn compress 64→16
        scene_raw = self.scene_proj(scene_seq) + pos64_sc + mod[MODALITY_SCENE]  # (B,64,384)
        scene = self.scene_compressor(scene_raw)                                  # (B,16,384)

        # Actions: project + pos64 + modality → cross-attn compress 64→16
        actions_raw = self.action_proj(action_seq) + pos64_act + mod[MODALITY_ACTION]  # (B,64,384)
        actions = self.action_compressor(actions_raw)                                   # (B,16,384)

        # Concat all tokens → unified context (all 16 tokens each)
        if self.use_audio and audio_seq is not None:
            audio   = self.audio_proj(audio_seq) + pos16 + mod[MODALITY_AUDIO]  # (B,16,384)
            context = torch.cat([radar, scene, audio, det, actions, state], dim=1)  # (B,81,384)
        else:
            context = torch.cat([radar, scene, det, actions, state], dim=1)          # (B,65,384)

        # Unified Transformer: all modalities attend to each other simultaneously
        context = self.unified_encoder(context)  # (B, 81, 384) or (B, 65, 384)

        # Register tokens cross-attend to the full fused context
        register_tokens = torch.cat([
            self.register_tokens_policy_mouse,
            self.register_tokens_policy_keys,
            self.register_tokens_value
        ], dim=1).expand(B, -1, -1)  # (B, 3, 384)

        fused = self.cross_attn(register_tokens, context, context)  # (B, 3, 384)

        mouse_embed = fused[:, 0, :]
        keys_embed  = fused[:, 1, :]
        value_embed = fused[:, 2, :]

        policy_keys = self.policy_keys_head(keys_embed).view(B, 20)
        value = self.value_head(value_embed)

        return mouse_embed, policy_keys, value


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TemporalCrossTransformer(use_audio=True).to(device)
    flow_head = FlowActionHead(context_dim=384).to(device)

    def count(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"modality_embedding: {count(model.modality_embedding):,} params")
    print(f"scene_compressor:   {count(model.scene_compressor):,} params")
    print(f"action_compressor:  {count(model.action_compressor):,} params")
    print(f"unified_encoder:    {count(model.unified_encoder):,} params")
    print(f"cross_attn:         {count(model.cross_attn):,} params")
    proj_total = (count(model.radar_proj) + count(model.scene_proj) +
                  count(model.audio_proj) + count(model.detection_proj) +
                  count(model.action_proj) + count(model.state_proj))
    print(f"projections:        {proj_total:,} params")
    heads_total = count(model.policy_keys_head) + count(model.value_head)
    print(f"output_heads:       {heads_total:,} params")
    print(f"TemporalTransformer TOTAL: {count(model):,} params  ({count(model)/1e6:.2f}M)")
    print(f"FlowActionHead:            {count(flow_head):,} params")

    B = 2
    radar   = torch.randn(B, 16, 512).to(device)   # 16 tokens
    scene   = torch.randn(B, 64, 512).to(device)   # 64 tokens → compressed to 16
    audio   = torch.randn(B, 16, 512).to(device)   # 16 tokens
    det     = torch.randn(B, 16, 100).to(device)   # 16 tokens
    actions = torch.randn(B, 64, 22).to(device)    # 64 tokens → compressed to 16
    state   = torch.randn(B, 95).to(device)
    gt_mouse = torch.randn(B, 2).to(device)

    mouse_embed, policy_keys, value = model(radar, scene, det, actions, state, audio)
    print(f"\nmouse_embed: {mouse_embed.shape}")   # (2, 384)
    print(f"policy_keys: {policy_keys.shape}")     # (2, 20)
    print(f"value:       {value.shape}")           # (2, 1)
    # After compression: 16+16+16+16+16+1 = 81 tokens

    loss = flow_head.compute_loss(mouse_embed, gt_mouse)
    print(f"Flow loss: {loss.item():.4f}")

    sampled = flow_head.sample(mouse_embed)
    print(f"Sampled action: {sampled.shape}")      # (2, 2)

    # Without audio: 16+16+16+16+1 = 65 tokens
    mouse_embed2, _, _ = model(radar, scene, det, actions, state, audio_seq=None)
    print(f"Without audio: {mouse_embed2.shape}")  # (2, 384)

    print("\nAll tests passed!")
