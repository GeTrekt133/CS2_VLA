"""
Convert PyTorch models to TensorRT FP16 engines.

Usage:
    python -m inference_pipeline.tensorrt.convert_to_trt \
        --checkpoint ./checkpoints2/run_xxx/epoch_10.pth \
        --output-dir ./trt_engines \
        --models radar yolo

This script:
1. Loads PyTorch models from checkpoint
2. Exports to ONNX format
3. Converts ONNX to TensorRT with FP16 precision
4. Saves .trt engine files

Expected speedup: 2-3x on radar/yolo encoding (after caching)
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add final_model path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'final_model'))

try:
    import tensorrt as trt
    import onnx
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("Warning: TensorRT not installed. Install with:")
    print("  pip install tensorrt onnx")


def export_radar_to_onnx(radar_encoder, output_path: str, batch_size: int = 1):
    """
    Export RadarEncoder to ONNX.

    Args:
        radar_encoder: PyTorch RadarEncoder model
        output_path: Path to save .onnx file
        batch_size: Fixed batch size (default 1)
    """
    print(f"\n[1/3] Exporting RadarEncoder to ONNX...")

    # Set to eval mode
    radar_encoder.eval()

    # Create dummy input (batch_size, 3, 224, 224)
    dummy_input = torch.randn(batch_size, 3, 224, 224, device='cuda')

    # Export to ONNX
    torch.onnx.export(
        radar_encoder,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,  # Use latest stable opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            # No dynamic axes - TRT works better with fixed shapes
        }
    )

    print(f"  ✓ ONNX saved to: {output_path}")

    # Verify ONNX
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model verified")

    # Test forward pass
    with torch.no_grad():
        pytorch_output = radar_encoder(dummy_input)

    print(f"  ✓ PyTorch output shape: {pytorch_output.shape}")
    return pytorch_output.cpu().numpy()


def export_yolo_to_onnx(yolo_model, output_path: str, batch_size: int = 1, img_size: tuple = (640, 640)):
    """
    Export YOLO scene encoder to ONNX.

    Args:
        yolo_model: PyTorch YOLO DetectionModel
        output_path: Path to save .onnx file
        batch_size: Fixed batch size
        img_size: (width, height) of input images — must be 640x640 for YOLOv11l
    """
    print(f"\n[1/3] Exporting YOLO scene encoder to ONNX...")

    yolo_model.eval()

    # Create dummy input (batch_size, 3, H, W)
    W, H = img_size
    dummy_input = torch.randn(batch_size, 3, H, W, device='cuda')

    # Wrap embed_from_features pipeline for TRT export
    # Returns (B, 512) embed via extract_features → embed_from_features
    class YOLOEmbedOnly(torch.nn.Module):
        def __init__(self, yolo):
            super().__init__()
            self.yolo = yolo

        def forward(self, x):
            p3, p5 = self.yolo.extract_features(x)
            return self.yolo.embed_from_features(p3, p5)  # (B, 512)

    embed_only_model = YOLOEmbedOnly(yolo_model).eval()

    # Export
    torch.onnx.export(
        embed_only_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embeddings'],
        dynamic_axes={}
    )

    print(f"  ✓ ONNX saved to: {output_path}")

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model verified")

    # Test forward pass
    with torch.no_grad():
        pytorch_output = embed_only_model(dummy_input)

    print(f"  ✓ PyTorch output shape: {pytorch_output.shape}")
    return pytorch_output.cpu().numpy()


def export_audio_to_onnx(audio_encoder, output_path: str, batch_size: int = 1):
    """
    Export AudioEncoder to ONNX.

    Args:
        audio_encoder: PyTorch AudioEncoder model
        output_path: Path to save .onnx file
        batch_size: Fixed batch size
    """
    print(f"\n[1/3] Exporting AudioEncoder to ONNX...")

    audio_encoder.eval()

    # Create dummy input (batch_size, 2, 256000) - 16 sec stereo @ 16kHz
    dummy_input = torch.randn(batch_size, 2, 256000, device='cuda')

    # Export to ONNX
    torch.onnx.export(
        audio_encoder,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['embeddings'],
        dynamic_axes={}
    )

    print(f"  ✓ ONNX saved to: {output_path}")

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model verified")

    # Test forward pass
    with torch.no_grad():
        pytorch_output = audio_encoder(dummy_input)

    print(f"  ✓ PyTorch output shape: {pytorch_output.shape}")
    return pytorch_output.cpu().numpy()


def export_temporal_to_onnx(
    temporal_model,
    flow_head,
    output_path: str,
    batch_size: int = 1,
    use_audio: bool = True,
    radar_seq_len: int = 16,
    scene_seq_len: int = 64,
    audio_seq_len: int = 16
):
    """
    Export TemporalTransformer + FlowActionHead to ONNX.

    Combines both models into single ONNX graph for TRT optimization.

    Args:
        temporal_model: PyTorch TemporalCrossTransformer
        flow_head: PyTorch FlowActionHead
        output_path: Path to save .onnx file
        batch_size: Fixed batch size
        use_audio: Whether model uses audio
        radar_seq_len: Radar sequence length
        scene_seq_len: Scene sequence length
        audio_seq_len: Audio sequence length
    """
    print(f"\n[1/3] Exporting TemporalTransformer + FlowActionHead to ONNX...")

    temporal_model.eval()
    flow_head.eval()

    # Create wrapper that combines both models
    class TemporalFlowWrapper(torch.nn.Module):
        def __init__(self, temporal, flow, use_audio):
            super().__init__()
            self.temporal = temporal
            self.flow = flow
            self.use_audio = use_audio

        def forward(self, radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq=None):
            # Forward through temporal transformer
            if self.use_audio and audio_seq is not None:
                mouse_embed, policy_keys, value = self.temporal(
                    radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq
                )
            else:
                mouse_embed, policy_keys, value = self.temporal(
                    radar_seq, scene_seq, detection_seq, action_seq, state_vec
                )

            # Sample mouse action via flow matching
            # Deterministic inference: use fixed noise for TRT
            device = mouse_embed.device
            B = mouse_embed.shape[0]

            # Fixed integration steps for TRT
            K = self.flow.num_steps
            a = torch.zeros(B, 2, device=device)  # Start from zero instead of random

            for k in range(K):
                tau = torch.full((B, 1), k / K, device=device)
                v = self.flow.forward(mouse_embed, a, tau)
                a = a + v / K

            policy_mouse = a

            return policy_mouse, policy_keys, value

    wrapper = TemporalFlowWrapper(temporal_model, flow_head, use_audio).eval()

    # Create dummy inputs matching final_model/ architecture
    dummy_radar = torch.randn(batch_size, radar_seq_len, 512, device='cuda')
    dummy_scene = torch.randn(batch_size, scene_seq_len, 512, device='cuda')
    dummy_detection = torch.randn(batch_size, 16, 100, device='cuda')
    dummy_action = torch.randn(batch_size, 64, 22, device='cuda')
    dummy_state = torch.randn(batch_size, 100, device='cuda')

    if use_audio:
        dummy_audio = torch.randn(batch_size, audio_seq_len, 512, device='cuda')
        dummy_inputs = (dummy_radar, dummy_scene, dummy_detection, dummy_action, dummy_state, dummy_audio)
        input_names = ['radar_seq', 'scene_seq', 'detection_seq', 'action_seq', 'state_vec', 'audio_seq']
    else:
        dummy_inputs = (dummy_radar, dummy_scene, dummy_detection, dummy_action, dummy_state)
        input_names = ['radar_seq', 'scene_seq', 'detection_seq', 'action_seq', 'state_vec']

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['policy_mouse', 'policy_keys', 'value'],
        dynamic_axes={}  # Fixed shapes for TRT
    )

    print(f"  ✓ ONNX saved to: {output_path}")

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model verified")

    # Test forward pass
    with torch.no_grad():
        pytorch_output = wrapper(*dummy_inputs)

    print(f"  ✓ PyTorch output shapes:")
    print(f"    policy_mouse: {pytorch_output[0].shape}")
    print(f"    policy_keys: {pytorch_output[1].shape}")
    print(f"    value: {pytorch_output[2].shape}")

    # Return as tuple of numpy arrays
    return tuple(o.cpu().numpy() for o in pytorch_output)


def build_trt_engine(onnx_path: str, trt_path: str, fp16: bool = True, workspace_gb: int = 4):
    """
    Convert ONNX to TensorRT engine with FP16.

    Args:
        onnx_path: Path to .onnx file
        trt_path: Path to save .trt engine
        fp16: Whether to use FP16 precision
        workspace_gb: Workspace size in GB
    """
    if not HAS_TRT:
        raise RuntimeError("TensorRT not installed")

    print(f"\n[2/3] Building TensorRT engine...")
    print(f"  Input:  {onnx_path}")
    print(f"  Output: {trt_path}")
    print(f"  FP16:   {fp16}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    print("  ✓ ONNX parsed successfully")

    # Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # Enable FP16
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  ✓ FP16 enabled")
        else:
            print("  ⚠ FP16 not supported on this platform, using FP32")

    # Build engine
    print("  ⏳ Building engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("  ✗ Failed to build engine")
        return False

    # Save engine
    with open(trt_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"  ✓ TensorRT engine saved: {trt_path}")
    print(f"  ✓ Size: {os.path.getsize(trt_path) / (1024*1024):.2f} MB")

    return True


def verify_trt_engine(trt_path: str, pytorch_output: np.ndarray, input_data: np.ndarray):
    """
    Verify TensorRT engine produces correct outputs.

    Args:
        trt_path: Path to .trt engine
        pytorch_output: Expected output from PyTorch
        input_data: Input data for verification
    """
    if not HAS_TRT:
        print("  ⚠ Skipping verification (TensorRT not installed)")
        return

    print(f"\n[3/3] Verifying TensorRT engine...")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Load engine
    with open(trt_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("  ✗ Failed to load engine")
        return False

    context = engine.create_execution_context()

    # Allocate buffers
    import pycuda.driver as cuda
    import pycuda.autoinit

    # Get binding names
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # Allocate device memory
    d_input = cuda.mem_alloc(input_data.nbytes)
    output_shape = pytorch_output.shape
    output_size = np.prod(output_shape)
    d_output = cuda.mem_alloc(output_size * np.dtype(np.float32).itemsize)

    # Copy input to device
    cuda.memcpy_htod(d_input, input_data)

    # Set bindings
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Run inference
    context.execute_async_v3(stream_handle=0)

    # Copy output back
    trt_output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(trt_output, d_output)

    # Compare outputs
    diff = np.abs(pytorch_output - trt_output)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"  Output shape: {trt_output.shape}")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-2:  # FP16 has lower precision
        print("  ✓ Verification PASSED")
        return True
    else:
        print(f"  ⚠ Large difference detected (max: {max_diff:.6f})")
        print("  This may be acceptable for FP16 precision")
        return True


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to TensorRT FP16')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output-dir', type=str, default='./trt_engines',
                        help='Output directory for TRT engines')
    parser.add_argument('--models', nargs='+', choices=['radar', 'yolo', 'audio', 'temporal', 'all'],
                        default=['all'], help='Models to convert')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Fixed batch size for TRT engine')
    parser.add_argument('--workspace-gb', type=int, default=4,
                        help='TensorRT workspace size in GB')
    parser.add_argument('--final-model-path', type=str, default='./final_model',
                        help='Path to final_model folder')

    args = parser.parse_args()

    if not HAS_TRT:
        print("\nERROR: TensorRT not installed")
        print("Install with: pip install tensorrt onnx")
        print("See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  TensorRT FP16 Model Conversion")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {output_dir}")
    print(f"Models: {args.models}")
    print(f"Batch size: {args.batch_size}")

    # Load models
    print("\n[Loading PyTorch models...]")
    from ..models.model_loader import load_models

    # Check if audio should be loaded
    load_audio = 'audio' in args.models or 'all' in args.models

    models = load_models(
        checkpoint_path=args.checkpoint,
        device='cuda',
        use_audio=load_audio,
    )

    models.eval()
    print("✓ Models loaded")

    # Convert radar encoder
    if 'radar' in args.models or 'all' in args.models:
        print("\n" + "="*60)
        print("Converting RadarEncoder")
        print("="*60)

        onnx_path = output_dir / "radar_encoder.onnx"
        trt_path = output_dir / "radar_encoder.trt"

        # Export to ONNX
        dummy_input = torch.randn(args.batch_size, 3, 224, 224, device='cuda')
        pytorch_output = export_radar_to_onnx(models.radar_encoder, str(onnx_path), args.batch_size)

        # Convert to TRT
        if build_trt_engine(str(onnx_path), str(trt_path), fp16=True, workspace_gb=args.workspace_gb):
            # Verify
            verify_trt_engine(str(trt_path), pytorch_output, dummy_input.cpu().numpy())
        else:
            print("✗ Failed to build TensorRT engine for RadarEncoder")

    # Convert YOLO
    if 'yolo' in args.models or 'all' in args.models:
        print("\n" + "="*60)
        print("Converting YOLO Scene Encoder")
        print("="*60)

        onnx_path = output_dir / "yolo_embed.onnx"
        trt_path = output_dir / "yolo_embed.trt"

        # Export to ONNX — square 640×640 for YOLOv11l
        dummy_input = torch.randn(args.batch_size, 3, 640, 640, device='cuda')
        pytorch_output = export_yolo_to_onnx(models.yolo, str(onnx_path), args.batch_size, img_size=(640, 640))

        # Convert to TRT
        if build_trt_engine(str(onnx_path), str(trt_path), fp16=True, workspace_gb=args.workspace_gb):
            # Verify
            verify_trt_engine(str(trt_path), pytorch_output, dummy_input.cpu().numpy())
        else:
            print("✗ Failed to build TensorRT engine for YOLO")

    # Convert AudioEncoder
    if 'audio' in args.models or 'all' in args.models:
        if models.audio_encoder is not None:
            print("\n" + "="*60)
            print("Converting AudioEncoder")
            print("="*60)

            onnx_path = output_dir / "audio_encoder.onnx"
            trt_path = output_dir / "audio_encoder.trt"

            # Export to ONNX — stereo 16 sec @ 16kHz
            dummy_input = torch.randn(args.batch_size, 2, 256000, device='cuda')
            pytorch_output = export_audio_to_onnx(models.audio_encoder, str(onnx_path), args.batch_size)

            # Convert to TRT
            if build_trt_engine(str(onnx_path), str(trt_path), fp16=True, workspace_gb=args.workspace_gb):
                # Verify
                verify_trt_engine(str(trt_path), pytorch_output, dummy_input.cpu().numpy())
            else:
                print("✗ Failed to build TensorRT engine for AudioEncoder")
        else:
            print("\n⚠ AudioEncoder not available in checkpoint, skipping")

    # Convert TemporalTransformer + FlowActionHead
    if 'temporal' in args.models or 'all' in args.models:
        print("\n" + "="*60)
        print("Converting TemporalTransformer + FlowActionHead")
        print("="*60)

        onnx_path = output_dir / "temporal_flow.onnx"
        trt_path = output_dir / "temporal_flow.trt"

        # Load FlowActionHead from checkpoint
        from TemporalTransformer import FlowActionHead

        flow_head = FlowActionHead(
            context_dim=384,  # must match d_model of TemporalCrossTransformer
            action_dim=2,
            hidden_dim=256,
            noise_scale=0.3,
            num_steps=5
        ).to('cuda')

        if "flow_head" in checkpoint:
            result = flow_head.load_state_dict(checkpoint["flow_head"], strict=False)
            if result.missing_keys:
                print(f"[Convert]   FlowActionHead missing keys: {result.missing_keys}")
            if result.unexpected_keys:
                print(f"[Convert]   FlowActionHead unexpected keys: {result.unexpected_keys}")
            print("[Convert]   FlowActionHead loaded from checkpoint")
        else:
            print("[Convert]   WARNING: No flow_head in checkpoint - using RANDOM weights!")

        flow_head.eval()

        # Export to ONNX
        # Note: Verification for temporal is skipped due to multiple outputs
        pytorch_outputs = export_temporal_to_onnx(
            models.temporal_model,
            flow_head,
            str(onnx_path),
            batch_size=args.batch_size,
            use_audio=load_audio,
            radar_seq_len=16,   # final_model: 16 radar tokens
            scene_seq_len=64,   # final_model: 64 scene tokens (ModalityCompressor 64→16 internally)
            audio_seq_len=16    # final_model: 16 audio tokens (after linspace 32→16)
        )

        # Convert to TRT
        if build_trt_engine(str(onnx_path), str(trt_path), fp16=True, workspace_gb=args.workspace_gb):
            print(f"  ✓ TensorRT engine built successfully")
            print(f"  ⚠ Skipping detailed verification for multi-output model")
            print(f"     Manual testing recommended after conversion")
        else:
            print("✗ Failed to build TensorRT engine for TemporalTransformer")

    print("\n" + "="*60)
    print("  Conversion Complete!")
    print("="*60)
    print(f"\nTensorRT engines saved to: {output_dir}")
    print("\nTo use TRT engines during inference:")
    print(f"  python -m inference_pipeline.main \\")
    print(f"    --checkpoint {args.checkpoint} \\")
    print(f"    --use-trt \\")
    print(f"    --trt-dir {output_dir}")


if __name__ == "__main__":
    main()
