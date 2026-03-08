"""Debug script to inspect checkpoint keys."""
import torch

ckpt_path = "D:/epoch_5.pth"
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print("\n=== Checkpoint temporal_model keys (first 30): ===")
temporal_keys = list(ckpt['temporal_model'].keys())
for k in temporal_keys[:30]:
    print(f"  {k}")
print(f"... ({len(temporal_keys)} total keys)")

# Now compare with what the model expects
import sys
sys.path.insert(0, str("audio_adaptation/src"))
from TemporalTransformer import TemporalCrossTransformer

print("\n=== Creating model to compare keys ===")
model = TemporalCrossTransformer(
    radar_dim=512,
    radar_seq=129,
    scene_dim=2048,
    scene_seq=16,
    audio_dim=512,
    audio_seq=60,
    detection_dim=100,
    detection_seq=1,
    actions_dim=22,
    actions_seq=16,
    state_dim=95,
    d_model=512,
    num_heads=8,
    depth=6,
    audio_depth=2,
    use_audio=True
)

model_keys = list(model.state_dict().keys())
print("\n=== Model expected keys (first 30): ===")
for k in model_keys[:30]:
    print(f"  {k}")
print(f"... ({len(model_keys)} total keys)")

# Find mismatches
ckpt_set = set(temporal_keys)
model_set = set(model_keys)

missing = model_set - ckpt_set  # Keys model needs but checkpoint doesn't have
unexpected = ckpt_set - model_set  # Keys checkpoint has but model doesn't need

print(f"\n=== Mismatch Summary ===")
print(f"Missing (model needs, checkpoint lacks): {len(missing)}")
print(f"Unexpected (checkpoint has, model doesn't need): {len(unexpected)}")

if len(missing) > 0:
    print("\nFirst 10 missing keys:")
    for k in list(missing)[:10]:
        print(f"  MODEL NEEDS: {k}")

if len(unexpected) > 0:
    print("\nFirst 10 unexpected keys:")
    for k in list(unexpected)[:10]:
        print(f"  CKPT HAS: {k}")
