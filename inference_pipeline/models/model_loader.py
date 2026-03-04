"""
Model loading utilities for the inference pipeline.
Loads RadarEncoder, YOLO (l-scale), StereoAudioEncoder, TemporalCrossTransformer,
and FlowActionHead from final_model/ checkpoints.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


@dataclass
class ModelBundle:
    """Container for all loaded models."""
    radar_encoder: nn.Module
    yolo: nn.Module
    temporal_model: nn.Module
    flow_head: nn.Module  # FlowActionHead for mouse prediction
    audio_encoder: Optional[nn.Module] = None

    device: str = "cuda"
    use_audio: bool = False

    def to(self, device: str) -> "ModelBundle":
        """Move all models to device."""
        self.device = device
        self.radar_encoder = self.radar_encoder.to(device)
        self.yolo = self.yolo.to(device)
        self.temporal_model = self.temporal_model.to(device)
        self.flow_head = self.flow_head.to(device)
        if self.audio_encoder is not None:
            self.audio_encoder = self.audio_encoder.to(device)
        return self

    def eval(self) -> "ModelBundle":
        """Set all models to eval mode."""
        self.radar_encoder.eval()
        self.yolo.eval()
        self.temporal_model.eval()
        self.flow_head.eval()
        if self.audio_encoder is not None:
            self.audio_encoder.eval()
        return self


def load_models(
    checkpoint_path: str,
    device: str = "cuda",
    use_audio: bool = True,
    use_trt: bool = False,
    trt_dir: Optional[str] = None,
    final_model_path: Optional[Path] = None,
) -> ModelBundle:
    """
    Load all models from a final_model/ checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        device: Device to load models to
        use_audio: Whether to load audio encoder
        use_trt: Whether to use TensorRT engines (if available)
        trt_dir: Directory containing TRT engines (auto-detected if None)
        final_model_path: Path to final_model/ folder (auto-detected if None)

    Returns:
        ModelBundle containing all loaded models
    """
    # Auto-detect paths
    base_path = Path(__file__).parent.parent.parent
    if final_model_path is None:
        final_model_path = base_path / "final_model"

    # Add final_model/ to sys.path for imports
    final_model_str = str(final_model_path)
    if final_model_str not in sys.path:
        sys.path.insert(0, final_model_str)

    print(f"[ModelLoader] Loading from {checkpoint_path}")
    print(f"[ModelLoader] Model source: {final_model_path}")
    print(f"[ModelLoader] Device: {device}")
    print(f"[ModelLoader] Audio: {use_audio}")
    print(f"[ModelLoader] TensorRT: {use_trt}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Log checkpoint contents
    print(f"[ModelLoader] Checkpoint keys: {list(checkpoint.keys())}")
    for key in ['radar_encoder', 'yolo', 'temporal_model', 'audio_encoder', 'flow_head']:
        if key in checkpoint:
            state_dict = checkpoint[key]
            n_params = sum(v.numel() for v in state_dict.values())
            print(f"[ModelLoader]   {key}: {len(state_dict)} tensors, {n_params:,} params")

    # Import model classes from final_model/
    from RadarEncoder import RadarEncoderEffB0
    from Yolo import DetectionModel
    from TemporalTransformer import TemporalCrossTransformer, FlowActionHead

    # === Load RadarEncoder ===
    # EfficientNet-B0 blocks 1-5, output 512-dim
    print("[ModelLoader] Loading RadarEncoder...")
    radar_encoder = RadarEncoderEffB0(
        pretrained=False,
        in_ch=3,
        out_ch=64,
        embed_dim=512
    )

    if "radar_encoder" in checkpoint:
        result = radar_encoder.load_state_dict(checkpoint["radar_encoder"], strict=False)
        if result.missing_keys:
            print(f"[ModelLoader]   RadarEncoder missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[ModelLoader]   RadarEncoder unexpected keys: {result.unexpected_keys}")
        print("[ModelLoader]   RadarEncoder loaded from checkpoint")
    else:
        print("[ModelLoader]   WARNING: No radar_encoder in checkpoint")

    # === Load YOLO (l-scale, nc=2) ===
    print("[ModelLoader] Loading YOLO (l-scale, nc=2)...")
    yolo_cfg = final_model_path / "yolo11.yaml"
    yolo = DetectionModel(cfg=str(yolo_cfg), ch=3, nc=2, scale='l')

    if "yolo" in checkpoint:
        result = yolo.load_state_dict(checkpoint["yolo"], strict=False)
        if result.missing_keys:
            print(f"[ModelLoader]   YOLO missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            print(f"[ModelLoader]   YOLO unexpected keys ({len(result.unexpected_keys)})")
        print("[ModelLoader]   YOLO loaded from checkpoint")
    else:
        print("[ModelLoader]   WARNING: No yolo in checkpoint")

    # Freeze YOLO backbone (only embed head is trainable, but at inference all is frozen)
    yolo.eval()

    # === Load StereoAudioEncoder (optional) ===
    audio_encoder = None
    if use_audio:
        try:
            from AudioEncoder import StereoAudioEncoder
            print("[ModelLoader] Loading StereoAudioEncoder (mn10_as, stereo)...")

            audio_encoder = StereoAudioEncoder(
                sample_rate=16000,
                n_mels=128,
                embed_dim=512,
                target_embeddings=32,
            )

            if "audio_encoder" in checkpoint:
                result = audio_encoder.load_state_dict(checkpoint["audio_encoder"], strict=False)
                if result.missing_keys:
                    print(f"[ModelLoader]   AudioEncoder missing keys: {result.missing_keys[:5]}...")
                if result.unexpected_keys:
                    print(f"[ModelLoader]   AudioEncoder unexpected keys: {result.unexpected_keys[:5]}...")
                print("[ModelLoader]   StereoAudioEncoder loaded from checkpoint")
            else:
                print("[ModelLoader]   WARNING: No audio_encoder in checkpoint - using RANDOM weights!")

        except ImportError as e:
            print(f"[ModelLoader]   StereoAudioEncoder not available: {e}")
            use_audio = False
            audio_encoder = None

    # === Load TemporalCrossTransformer ===
    # Uses defaults from final_model/TemporalTransformer.py:
    #   d_model=384, depth=6, scene_seq_len=64, actions_seq_len=64, state_dim=100
    print("[ModelLoader] Loading TemporalCrossTransformer (d=384)...")

    temporal_model = TemporalCrossTransformer(use_audio=use_audio)

    if "temporal_model" in checkpoint:
        state_dict = checkpoint["temporal_model"]

        # Check if trained with DataParallel (keys start with "module.")
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith("module."):
            print("[ModelLoader]   Detected DataParallel checkpoint, stripping 'module.' prefix...")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        result = temporal_model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"[ModelLoader]   TemporalTransformer missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            print(f"[ModelLoader]   TemporalTransformer unexpected keys ({len(result.unexpected_keys)})")
        print("[ModelLoader]   TemporalTransformer loaded from checkpoint")
    else:
        print("[ModelLoader]   WARNING: No temporal_model in checkpoint - using RANDOM weights!")

    # === Load FlowActionHead ===
    # context_dim MUST match d_model=384
    print("[ModelLoader] Loading FlowActionHead (context_dim=384)...")

    flow_head = FlowActionHead(
        context_dim=384,
        action_dim=2,
        hidden_dim=256,
        noise_scale=0.3,
        num_steps=5
    )

    if "flow_head" in checkpoint:
        result = flow_head.load_state_dict(checkpoint["flow_head"], strict=False)
        if result.missing_keys:
            print(f"[ModelLoader]   FlowActionHead missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[ModelLoader]   FlowActionHead unexpected keys: {result.unexpected_keys}")
        print("[ModelLoader]   FlowActionHead loaded from checkpoint")
    else:
        print("[ModelLoader]   WARNING: No flow_head in checkpoint - using RANDOM weights!")

    # === Load TensorRT engines (if requested) ===
    if use_trt:
        print("[ModelLoader] Loading TensorRT engines...")

        if trt_dir is None:
            trt_dir = str(base_path / "trt_engines")

        try:
            from ..tensorrt.trt_wrapper import load_trt_models

            trt_models = load_trt_models(trt_dir, device, use_audio=use_audio)

            if trt_models.get('radar_encoder') is not None:
                print("[ModelLoader]   Replacing RadarEncoder with TRT version")
                radar_encoder = trt_models['radar_encoder']

            if trt_models.get('yolo_embed') is not None:
                print("[ModelLoader]   Replacing YOLO embed with TRT version")
                yolo.trt_embed = trt_models['yolo_embed']

            if trt_models.get('audio_encoder') is not None and audio_encoder is not None:
                print("[ModelLoader]   Replacing AudioEncoder with TRT version")
                audio_encoder = trt_models['audio_encoder']

            if trt_models.get('temporal_flow') is not None:
                print("[ModelLoader]   Replacing TemporalTransformer + FlowActionHead with TRT version")

                class TRTTemporalFlowWrapper:
                    def __init__(self, trt_temporal_flow):
                        self.trt_engine = trt_temporal_flow
                        self.device = trt_temporal_flow.device

                    def __call__(self, radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq=None):
                        policy_mouse, policy_keys, value = self.trt_engine(
                            radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq
                        )
                        return policy_mouse, policy_keys, value

                    def forward(self, *args, **kwargs):
                        return self(*args, **kwargs)

                    def eval(self):
                        return self

                    def to(self, device):
                        self.device = device
                        return self

                trt_wrapper = TRTTemporalFlowWrapper(trt_models['temporal_flow'])
                temporal_model = trt_wrapper
                flow_head = trt_wrapper

        except ImportError as e:
            print(f"[ModelLoader]   TensorRT not available: {e}")
            print("[ModelLoader]   Falling back to PyTorch models")
        except FileNotFoundError as e:
            print(f"[ModelLoader]   TRT engines not found: {e}")
            print("[ModelLoader]   Run conversion first:")
            print(f"[ModelLoader]     python -m inference_pipeline.tensorrt.convert_to_trt --checkpoint {checkpoint_path}")

    # Create bundle
    bundle = ModelBundle(
        radar_encoder=radar_encoder,
        yolo=yolo,
        temporal_model=temporal_model,
        flow_head=flow_head,
        audio_encoder=audio_encoder,
        device=device,
        use_audio=use_audio
    )

    # Move to device and set eval mode
    bundle.to(device).eval()

    # Count parameters
    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [radar_encoder, yolo, temporal_model]
        if m is not None and hasattr(m, 'parameters')
    )
    if audio_encoder is not None and hasattr(audio_encoder, 'parameters'):
        total_params += sum(p.numel() for p in audio_encoder.parameters())
    if hasattr(flow_head, 'parameters'):
        total_params += sum(p.numel() for p in flow_head.parameters())

    print(f"[ModelLoader] Total parameters: {total_params:,}")
    print("[ModelLoader] Models loaded successfully")

    return bundle


def load_buy_agent(model_path: str = "./buy_models/buy_predictor_v1"):
    """Load buy prediction agent."""
    try:
        base_path = Path(__file__).parent.parent.parent
        buy_path = base_path / "buy_prediction"
        if str(buy_path) not in sys.path:
            sys.path.insert(0, str(buy_path))

        from inference import BuyAgent
        agent = BuyAgent(model_path)
        print(f"[ModelLoader] Buy agent loaded from {model_path}")
        return agent
    except Exception as e:
        print(f"[ModelLoader] Buy agent not available: {e}")
        return None
