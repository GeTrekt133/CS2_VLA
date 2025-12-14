import torch
import torch.nn as nn
import math


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
    def __init__(self,
                 radar_dim=512, radar_seq=192,
                 scene_dim=2048, scene_seq=16,
                 detection_dim=100, detection_seq=1,
                 actions_dim=24, actions_seq=192,
                 state_dim=95,
                 d_model=512, num_heads=8, depth=6,
                 ff_mult=4, dropout=0.1):
        super().__init__()

        # === Projections ===
        self.radar_proj = nn.Linear(radar_dim, d_model)
        self.scene_proj = nn.Linear(scene_dim, d_model)
        self.detection_proj = nn.Linear(detection_dim, d_model)
        self.action_proj = nn.Linear(actions_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # === Positional encodings (sinusoidal) ===
        self.register_buffer("pos_embed_radar", sinusoidal_position_encoding(radar_seq, d_model, torch.device("cpu")))
        self.register_buffer("pos_embed_scene", sinusoidal_position_encoding(scene_seq, d_model, torch.device("cpu")))
        self.register_buffer("pos_embed_action", sinusoidal_position_encoding(actions_seq, d_model, torch.device("cpu")))

        # === Register tokens for different heads ===
        self.register_tokens_policy_mouse = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_tokens_policy_keys = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_tokens_value = nn.Parameter(torch.randn(1, 1, d_model))

        # === Transformer encoders ===
        self.radar_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True),
            num_layers=depth
        )
        self.scene_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True),
            num_layers=depth
        )

        # === Cross attention ===
        self.cross_attn = CrossAttentionBlock(d_model, num_heads=num_heads, ff_mult=ff_mult, dropout=dropout)

        # === Output heads ===
        self.policy_mouse_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 8)  # регрессия x,y мыши
        )
        self.policy_keys_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 88)  # 10 клавиш по 4 тика например
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, radar_seq, scene_seq, detection_seq, action_seq, state_vec):
        B = radar_seq.shape[0]

        # === Apply positional encoding ===
        radar = self.radar_proj(radar_seq) + self.pos_embed_radar[:, :radar_seq.shape[1], :].to(radar_seq.device)
        # print('radar - ', radar.shape)
        scene = self.scene_proj(scene_seq) + self.pos_embed_scene[:, :scene_seq.shape[1], :].to(scene_seq.device)
        # print('scene - ', scene.shape)
        actions = self.action_proj(action_seq) + self.pos_embed_action[:, :action_seq.shape[1], :].to(action_seq.device)
        # print('actions - ', actions.shape)
        state = self.state_proj(state_vec).unsqueeze(1)
        # print('state - ', state.shape)
        detections = self.detection_proj(detection_seq)

        # === Encode radar and scene sequences ===
        radar = self.radar_encoder(radar)
        scene = self.scene_encoder(scene)

        # === Expand register tokens for batch ===
        register_tokens = torch.cat([
            self.register_tokens_policy_mouse,
            self.register_tokens_policy_keys,
            self.register_tokens_value
        ], dim=1).expand(B, -1, -1)  # (B, 3, d_model)

        # === Prepare context: radar + detections + actions + state ===
        # print(radar.shape, scene.shape, detections.shape, actions.shape, state.shape)
        context = torch.cat([radar, scene, detections, actions, state], dim=1)
        # print(context.shape)

        # === Cross-attention: queries = register_tokens, key/value = context ===
        fused_tokens = self.cross_attn(register_tokens, context, context)  # (B, 3, d_model)
        # print(fused_tokens.shape)
        # === Extract each token ===
        policy_mouse_embed = fused_tokens[:, 0, :]
        policy_keys_embed = fused_tokens[:, 1, :]
        value_embed = fused_tokens[:, 2, :]

        # === Forward through heads ===
        policy_mouse = self.policy_mouse_head(policy_mouse_embed)
        policy_mouse = policy_mouse.view(B, 4, 2)
        policy_keys = self.policy_keys_head(policy_keys_embed)
        policy_keys = policy_keys.view(B, 4, 22)
        value = self.value_head(value_embed)

        return policy_mouse, policy_keys, value



if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    model = TemporalCrossTransformer().to('cuda')
    for i in range(100):
        radar_seq = torch.randn(1, 180, 512).to('cuda')
        scene_seq = torch.randn(1, 15, 2048).to('cuda')
        detection_seq = torch.randn(1, 4, 128).to('cuda')
        action_seq = torch.randn(1, 32, 64).to('cuda')
        state_vec = torch.randn(1, 16).to('cuda')
        policy_mouse, policy_keys, value = model(radar_seq, scene_seq, detection_seq, action_seq, state_vec)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(i)
    print(f"Обучаемых параметров: {total_params:,}")
