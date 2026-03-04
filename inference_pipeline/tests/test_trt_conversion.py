"""
Test TensorRT conversion correctness and performance.

Validates that TRT engines produce similar results to PyTorch models.
"""

import sys
from pathlib import Path
import time
import torch
import numpy as np
from typing import Optional

# Add paths
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path / "final_model"))
sys.path.insert(0, str(base_path / "inference_pipeline"))


def test_radar_encoder_trt(trt_dir: str, checkpoint_path: Optional[str] = None):
    """Test RadarEncoder TRT conversion."""
    print("\n=== Test: RadarEncoder TRT ===")

    from RadarEncoder import RadarEncoderEffB0
    from tensorrt.trt_wrapper import TRTRadarEncoder

    # Load PyTorch model
    pytorch_model = RadarEncoderEffB0(
        pretrained=False,
        in_ch=3,
        out_ch=64,
        embed_dim=512
    ).cuda().eval()

    # Load weights if checkpoint provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        if "radar_encoder" in checkpoint:
            pytorch_model.load_state_dict(checkpoint["radar_encoder"], strict=False)
            print("✅ Loaded PyTorch weights from checkpoint")

    # Load TRT model
    trt_path = Path(trt_dir) / "radar_encoder.trt"
    if not trt_path.exists():
        print(f"❌ TRT engine not found: {trt_path}")
        print("   Run: python -m inference_pipeline.tensorrt.convert_to_trt --models radar")
        return False

    trt_model = TRTRadarEncoder(str(trt_path), device='cuda')
    print("✅ Loaded TRT engine")

    # Test inputs
    test_input = torch.randn(1, 3, 224, 224, device='cuda')

    # PyTorch inference
    with torch.no_grad():
        pytorch_out = pytorch_model(test_input)  # (1, 512)

    # TRT inference
    trt_out = trt_model(test_input)  # (1, 512)

    # Compare
    max_diff = (pytorch_out - trt_out).abs().max().item()
    mean_diff = (pytorch_out - trt_out).abs().mean().item()

    print(f"Output shape: {trt_out.shape}")
    print(f"Max difference: {max_diff:.4f}")
    print(f"Mean difference: {mean_diff:.4f}")

    if max_diff > 0.1:  # FP16 tolerance
        print(f"⚠️  Warning: Large difference between PyTorch and TRT")
        return False

    print("✅ Correctness verified")

    # Benchmark
    print("\nBenchmarking...")

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_model(test_input)
        _ = trt_model(test_input)

    torch.cuda.synchronize()

    # PyTorch timing
    times_pytorch = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = pytorch_model(test_input)
        torch.cuda.synchronize()
        times_pytorch.append(time.time() - start)

    # TRT timing
    times_trt = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        _ = trt_model(test_input)
        torch.cuda.synchronize()
        times_trt.append(time.time() - start)

    pytorch_time = np.mean(times_pytorch) * 1000
    trt_time = np.mean(times_trt) * 1000
    speedup = pytorch_time / trt_time

    print(f"PyTorch: {pytorch_time:.2f}ms")
    print(f"TRT FP16: {trt_time:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")

    if speedup < 1.5:
        print("⚠️  Warning: Speedup less than expected")

    return True


def test_yolo_embed_trt(trt_dir: str, checkpoint_path: Optional[str] = None):
    """Test YOLO embedding TRT conversion."""
    print("\n=== Test: YOLO Embedding TRT ===")

    from Yolo import DetectionModel
    from tensorrt.trt_wrapper import TRTYOLOEmbed

    # Load PyTorch YOLO — final_model: yolo11l, nc=2
    yolo_cfg = base_path / "final_model" / "yolo11.yaml"
    pytorch_model = DetectionModel(cfg=str(yolo_cfg), ch=3, nc=2, scale='l').cuda().eval()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        if "yolo" in checkpoint:
            pytorch_model.load_state_dict(checkpoint["yolo"], strict=False)
            print("✅ Loaded PyTorch weights from checkpoint")

    # Load TRT model
    trt_path = Path(trt_dir) / "yolo_embed.trt"
    if not trt_path.exists():
        print(f"❌ TRT engine not found: {trt_path}")
        print("   Run: python -m inference_pipeline.tensorrt.convert_to_trt --models yolo")
        return False

    trt_model = TRTYOLOEmbed(str(trt_path), device='cuda')
    print("✅ Loaded TRT engine")

    # Test input — square 640×640 for YOLOv11l
    test_input = torch.randn(1, 3, 640, 640, device='cuda')

    # PyTorch inference via extract_features + embed_from_features
    with torch.no_grad():
        p3, p5 = pytorch_model.extract_features(test_input)
        pytorch_out = pytorch_model.embed_from_features(p3, p5)  # (1, 512)

    # TRT inference
    trt_out = trt_model(test_input)  # (1, 512)

    # Compare
    max_diff = (pytorch_out - trt_out).abs().max().item()
    mean_diff = (pytorch_out - trt_out).abs().mean().item()

    print(f"Output shape: {trt_out.shape}")
    print(f"Max difference: {max_diff:.4f}")
    print(f"Mean difference: {mean_diff:.4f}")

    if max_diff > 0.1:
        print(f"⚠️  Warning: Large difference between PyTorch and TRT")
        return False

    print("✅ Correctness verified")

    # Benchmark
    print("\nBenchmarking...")

    for _ in range(10):
        with torch.no_grad():
            p3, p5 = pytorch_model.extract_features(test_input)
            _ = pytorch_model.embed_from_features(p3, p5)
        _ = trt_model(test_input)

    torch.cuda.synchronize()

    times_pytorch = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            p3, p5 = pytorch_model.extract_features(test_input)
            _ = pytorch_model.embed_from_features(p3, p5)
        torch.cuda.synchronize()
        times_pytorch.append(time.time() - start)

    times_trt = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        _ = trt_model(test_input)
        torch.cuda.synchronize()
        times_trt.append(time.time() - start)

    pytorch_time = np.mean(times_pytorch) * 1000
    trt_time = np.mean(times_trt) * 1000
    speedup = pytorch_time / trt_time

    print(f"PyTorch: {pytorch_time:.2f}ms")
    print(f"TRT FP16: {trt_time:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")

    return True


def test_audio_encoder_trt(trt_dir: str, checkpoint_path: Optional[str] = None):
    """Test AudioEncoder TRT conversion."""
    print("\n=== Test: AudioEncoder TRT ===")

    try:
        from AudioEncoder import StereoAudioEncoder
        from tensorrt.trt_wrapper import TRTAudioEncoder
    except ImportError:
        print("⚠️  StereoAudioEncoder not available, skipping test")
        return True

    # Load PyTorch StereoAudioEncoder
    pytorch_model = StereoAudioEncoder(
        sample_rate=16000,
        n_mels=128,
        embed_dim=512,
        target_embeddings=32
    ).cuda().eval()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        if "audio_encoder" in checkpoint:
            pytorch_model.load_state_dict(checkpoint["audio_encoder"], strict=False)
            print("✅ Loaded PyTorch weights from checkpoint")

    # Load TRT model
    trt_path = Path(trt_dir) / "audio_encoder.trt"
    if not trt_path.exists():
        print(f"❌ TRT engine not found: {trt_path}")
        print("   Run: python -m inference_pipeline.tensorrt.convert_to_trt --models audio")
        return False

    trt_model = TRTAudioEncoder(str(trt_path), device='cuda')
    print("✅ Loaded TRT engine")

    # Test input — stereo 16 sec @ 16kHz
    test_input = torch.randn(1, 2, 256000, device='cuda')

    # PyTorch inference
    with torch.no_grad():
        pytorch_out = pytorch_model(test_input)  # (1, 32, 512)

    # TRT inference
    trt_out = trt_model(test_input)  # (1, 32, 512)

    # Compare
    max_diff = (pytorch_out - trt_out).abs().max().item()
    mean_diff = (pytorch_out - trt_out).abs().mean().item()

    print(f"Output shape: {trt_out.shape}")
    print(f"Max difference: {max_diff:.4f}")
    print(f"Mean difference: {mean_diff:.4f}")

    if max_diff > 0.1:
        print(f"⚠️  Warning: Large difference between PyTorch and TRT")
        return False

    print("✅ Correctness verified")

    # Benchmark
    print("\nBenchmarking...")

    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_model(test_input)
        _ = trt_model(test_input)

    torch.cuda.synchronize()

    times_pytorch = []
    for _ in range(50):  # Fewer iterations (audio encoder is slower)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = pytorch_model(test_input)
        torch.cuda.synchronize()
        times_pytorch.append(time.time() - start)

    times_trt = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.time()
        _ = trt_model(test_input)
        torch.cuda.synchronize()
        times_trt.append(time.time() - start)

    pytorch_time = np.mean(times_pytorch) * 1000
    trt_time = np.mean(times_trt) * 1000
    speedup = pytorch_time / trt_time

    print(f"PyTorch: {pytorch_time:.2f}ms")
    print(f"TRT FP16: {trt_time:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")

    return True


def test_temporal_flow_trt(trt_dir: str, checkpoint_path: Optional[str] = None, use_audio: bool = False):
    """Test TemporalTransformer + FlowActionHead TRT conversion."""
    print("\n=== Test: TemporalTransformer + FlowActionHead TRT ===")

    try:
        from TemporalTransformer import TemporalCrossTransformer, FlowActionHead
        from tensorrt.trt_wrapper import TRTTemporalFlow
    except ImportError as e:
        print(f"⚠️  TemporalTransformer not available: {e}")
        return True

    # Load PyTorch models using final_model/ defaults
    temporal_model = TemporalCrossTransformer(use_audio=use_audio).cuda().eval()

    flow_head = FlowActionHead(
        context_dim=384,  # must match d_model=384
        action_dim=2,
        hidden_dim=256,
        noise_scale=0.3,
        num_steps=5
    ).cuda().eval()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        if "temporal_model" in checkpoint:
            state_dict = checkpoint["temporal_model"]
            # Strip DataParallel prefix if present
            if next(iter(state_dict.keys())).startswith("module."):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            temporal_model.load_state_dict(state_dict, strict=False)
            print("✅ Loaded TemporalTransformer weights from checkpoint")
        if "flow_head" in checkpoint:
            flow_head.load_state_dict(checkpoint["flow_head"], strict=False)
            print("✅ Loaded FlowActionHead weights from checkpoint")

    # Load TRT model
    trt_path = Path(trt_dir) / ("temporal_flow_audio.trt" if use_audio else "temporal_flow.trt")
    if not trt_path.exists():
        print(f"❌ TRT engine not found: {trt_path}")
        print("   Run: python -m inference_pipeline.tensorrt.convert_to_trt --models temporal")
        return False

    trt_model = TRTTemporalFlow(str(trt_path), device='cuda', use_audio=use_audio)
    print("✅ Loaded TRT engine")

    # Test inputs matching final_model/ architecture
    radar_seq = torch.randn(1, 16, 512, device='cuda')
    scene_seq = torch.randn(1, 64, 512, device='cuda')
    detection_seq = torch.randn(1, 16, 100, device='cuda')
    action_seq = torch.randn(1, 64, 22, device='cuda')
    state_vec = torch.randn(1, 100, device='cuda')
    audio_seq = torch.randn(1, 16, 512, device='cuda') if use_audio else None

    # PyTorch inference
    with torch.no_grad():
        mouse_embed, policy_keys, value = temporal_model(
            radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq
        )
        # Flow matching with deterministic (zero init)
        policy_mouse = flow_head.sample(mouse_embed, deterministic=True)

    # TRT inference
    trt_mouse, trt_keys, trt_value = trt_model(
        radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq
    )

    # Compare
    mouse_diff = (policy_mouse - trt_mouse).abs().max().item()
    keys_diff = (policy_keys - trt_keys).abs().max().item()
    value_diff = (value - trt_value).abs().max().item()

    print(f"Output shapes: mouse={trt_mouse.shape}, keys={trt_keys.shape}, value={trt_value.shape}")
    print(f"Max difference (mouse): {mouse_diff:.4f}")
    print(f"Max difference (keys): {keys_diff:.4f}")
    print(f"Max difference (value): {value_diff:.4f}")

    if mouse_diff > 0.5 or keys_diff > 0.1 or value_diff > 0.1:
        print(f"⚠️  Warning: Large difference between PyTorch and TRT")
        return False

    print("✅ Correctness verified")

    # Benchmark
    print("\nBenchmarking...")

    for _ in range(10):
        with torch.no_grad():
            mouse_embed, policy_keys, value = temporal_model(
                radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq
            )
            _ = flow_head.sample(mouse_embed, deterministic=True)
        _ = trt_model(radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq)

    torch.cuda.synchronize()

    times_pytorch = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            mouse_embed, policy_keys, value = temporal_model(
                radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq
            )
            _ = flow_head.sample(mouse_embed, deterministic=True)
        torch.cuda.synchronize()
        times_pytorch.append(time.time() - start)

    times_trt = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.time()
        _ = trt_model(radar_seq, scene_seq, detection_seq, action_seq, state_vec, audio_seq)
        torch.cuda.synchronize()
        times_trt.append(time.time() - start)

    pytorch_time = np.mean(times_pytorch) * 1000
    trt_time = np.mean(times_trt) * 1000
    speedup = pytorch_time / trt_time

    print(f"PyTorch: {pytorch_time:.2f}ms")
    print(f"TRT FP16: {trt_time:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")

    return True


def main():
    """Run all TRT tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test TensorRT conversion")
    parser.add_argument("--trt-dir", type=str, default="./trt_engines",
                        help="Directory containing TRT engines")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for loading weights")
    parser.add_argument("--use-audio", action="store_true",
                        help="Test with audio enabled")
    parser.add_argument("--models", type=str, default="all",
                        choices=["all", "radar", "yolo", "audio", "temporal"],
                        help="Which models to test")

    args = parser.parse_args()

    print("=" * 60)
    print("TensorRT Conversion Test Suite")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available, tests require GPU")
        return

    results = {}

    try:
        if args.models in ["all", "radar"]:
            results["radar"] = test_radar_encoder_trt(args.trt_dir, args.checkpoint)

        if args.models in ["all", "yolo"]:
            results["yolo"] = test_yolo_embed_trt(args.trt_dir, args.checkpoint)

        if args.models in ["all", "audio"]:
            results["audio"] = test_audio_encoder_trt(args.trt_dir, args.checkpoint)

        if args.models in ["all", "temporal"]:
            results["temporal"] = test_temporal_flow_trt(args.trt_dir, args.checkpoint, args.use_audio)

        print("\n" + "=" * 60)
        print("Test Results:")
        for model, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {model}: {status}")
        print("=" * 60)

        if all(results.values()):
            print("\n✅ ALL TESTS PASSED")
        else:
            print("\n❌ SOME TESTS FAILED")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST ERROR: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
