"""
Setup validation script - checks if all requirements are met for optimizations.

Validates:
- CUDA availability
- TensorRT installation
- Model checkpoints
- TRT engines
- GPU capabilities (FP16 support)
- Memory requirements
"""

import sys
from pathlib import Path
import subprocess

# Add paths
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path / "src"))
sys.path.insert(0, str(base_path / "audio_adaptation" / "src"))


def check_cuda():
    """Check CUDA availability."""
    print("\n" + "="*60)
    print("Checking CUDA...")
    print("="*60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            print("   Install CUDA from: https://developer.nvidia.com/cuda-downloads")
            return False

        device_count = torch.cuda.device_count()
        print(f"✅ CUDA available")
        print(f"   Devices: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   Device {i}: {props.name}")
            print(f"     Compute Capability: {props.major}.{props.minor}")
            print(f"     Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"     Multi-Processors: {props.multi_processor_count}")

            # Check FP16 support
            if props.major >= 6:  # Pascal (6.x) or newer
                print(f"     FP16 Support: ✅ YES (Compute >= 6.0)")
            else:
                print(f"     FP16 Support: ❌ NO (need Compute >= 6.0)")
                print(f"       TensorRT FP16 will not work on this GPU")

        return True

    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False


def check_tensorrt():
    """Check TensorRT installation."""
    print("\n" + "="*60)
    print("Checking TensorRT...")
    print("="*60)

    try:
        import tensorrt as trt
        print(f"✅ TensorRT installed")
        print(f"   Version: {trt.__version__}")

        # Check if pycuda is available
        try:
            import pycuda
            import pycuda.driver as cuda
            print(f"✅ PyCUDA installed")
            print(f"   Version: {pycuda.VERSION_TEXT}")
        except ImportError:
            print("❌ PyCUDA not installed")
            print("   Install: pip install pycuda")
            return False

        return True

    except ImportError:
        print("❌ TensorRT not installed")
        print("   Install: pip install tensorrt pycuda")
        print("   Or download from: https://developer.nvidia.com/tensorrt")
        return False


def check_checkpoint(checkpoint_path: str):
    """Check if checkpoint exists and contains required keys."""
    print("\n" + "="*60)
    print("Checking checkpoint...")
    print("="*60)

    import torch

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    print(f"✅ Checkpoint found: {checkpoint_path}")
    print(f"   Size: {checkpoint_path.stat().st_size / 1024**2:.2f} MB")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        required_keys = ['radar_encoder', 'yolo', 'temporal_model', 'flow_head']
        optional_keys = ['audio_encoder', 'optimizer']

        print("\n   Required components:")
        all_required = True
        for key in required_keys:
            if key in checkpoint:
                n_params = sum(v.numel() for v in checkpoint[key].values())
                print(f"     ✅ {key}: {n_params:,} parameters")
            else:
                print(f"     ❌ {key}: MISSING")
                all_required = False

        print("\n   Optional components:")
        for key in optional_keys:
            if key in checkpoint:
                if key == 'optimizer':
                    print(f"     ✅ {key}")
                else:
                    n_params = sum(v.numel() for v in checkpoint[key].values())
                    print(f"     ✅ {key}: {n_params:,} parameters")
            else:
                print(f"     ⚠️  {key}: not found")

        return all_required

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False


def check_trt_engines(trt_dir: str):
    """Check if TRT engines exist."""
    print("\n" + "="*60)
    print("Checking TRT engines...")
    print("="*60)

    trt_dir = Path(trt_dir)

    if not trt_dir.exists():
        print(f"❌ TRT directory not found: {trt_dir}")
        print("   Run conversion first:")
        print("   python -m inference_pipeline.tensorrt.convert_to_trt --checkpoint <path> --models all")
        return False

    print(f"✅ TRT directory found: {trt_dir}")

    expected_engines = [
        'radar_encoder.trt',
        'yolo_embed.trt',
        'temporal_flow.trt',
        'audio_encoder.trt',  # optional
    ]

    found_engines = []
    missing_engines = []

    for engine_name in expected_engines:
        engine_path = trt_dir / engine_name
        if engine_path.exists():
            size = engine_path.stat().st_size / 1024**2
            print(f"   ✅ {engine_name}: {size:.2f} MB")
            found_engines.append(engine_name)
        else:
            if engine_name == 'audio_encoder.trt':
                print(f"   ⚠️  {engine_name}: not found (optional)")
            else:
                print(f"   ❌ {engine_name}: MISSING")
                missing_engines.append(engine_name)

    if missing_engines:
        print("\n   To generate missing engines:")
        print(f"   python -m inference_pipeline.tensorrt.convert_to_trt \\")
        print(f"       --checkpoint <path> \\")
        print(f"       --output-dir {trt_dir} \\")
        print(f"       --models all")
        return False

    return True


def check_gpu_memory():
    """Check available GPU memory."""
    print("\n" + "="*60)
    print("Checking GPU memory...")
    print("="*60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False

        torch.cuda.empty_cache()

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1024**3

            # Get current usage
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = total_mem - reserved

            print(f"\n   Device {i}: {props.name}")
            print(f"     Total: {total_mem:.2f} GB")
            print(f"     Allocated: {allocated:.2f} GB")
            print(f"     Reserved: {reserved:.2f} GB")
            print(f"     Free: {free:.2f} GB")

            # Check requirements
            min_required = 3.0  # GB for TRT FP16
            recommended = 4.0   # GB for comfortable operation

            if total_mem < min_required:
                print(f"     ❌ Insufficient memory (need {min_required} GB minimum)")
            elif total_mem < recommended:
                print(f"     ⚠️  Marginal memory (recommended {recommended}+ GB)")
            else:
                print(f"     ✅ Sufficient memory")

        return True

    except Exception as e:
        print(f"❌ Failed to check GPU memory: {e}")
        return False


def check_model_imports():
    """Check if all model classes can be imported."""
    print("\n" + "="*60)
    print("Checking model imports...")
    print("="*60)

    imports_ok = True

    # Required imports
    required_modules = [
        ('RadarEncoder', 'RadarEncoderEffB0'),
        ('Yolo', 'DetectionModel'),
        ('TemporalTransformer', 'TemporalCrossTransformer'),
        ('TemporalTransformer', 'FlowActionHead'),
    ]

    for module_name, class_name in required_modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"   ✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"   ❌ {module_name}.{class_name}: {e}")
            imports_ok = False

    # Optional imports
    optional_modules = [
        ('AudioEncoder', 'AudioEncoder'),
    ]

    print("\n   Optional components:")
    for module_name, class_name in optional_modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"   ✅ {module_name}.{class_name}")
        except ImportError:
            print(f"   ⚠️  {module_name}.{class_name}: not available (audio disabled)")

    return imports_ok


def run_quick_test():
    """Run quick functionality test."""
    print("\n" + "="*60)
    print("Running quick functionality test...")
    print("="*60)

    try:
        import torch
        from inference.embedding_cache import GPUEmbeddingCache

        # Test cache creation
        cache = GPUEmbeddingCache(capacity=32, emb_dim=512, device='cuda')
        print("   ✅ Embedding cache creation")

        # Test mock encoding
        def mock_encoder(x):
            return torch.randn(1, 512, device='cuda')

        for _ in range(10):
            seq = cache.add_and_get_sequence(
                torch.randn(1, 3, 224, 224, device='cuda'),
                mock_encoder,
                seq_length=8
            )

        print("   ✅ Cache operations")
        print(f"   ✅ Memory: {cache.get_stats()['memory_mb']:.2f} MB")

        return True

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def main():
    """Run all validation checks."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate inference pipeline setup")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (optional)")
    parser.add_argument("--trt-dir", type=str, default="./trt_engines",
                        help="Directory containing TRT engines")

    args = parser.parse_args()

    print("="*60)
    print("INFERENCE PIPELINE SETUP VALIDATION")
    print("="*60)

    results = {}

    # Core checks
    results['CUDA'] = check_cuda()
    results['GPU Memory'] = check_gpu_memory()
    results['Model Imports'] = check_model_imports()

    # Optional checks
    if args.checkpoint:
        results['Checkpoint'] = check_checkpoint(args.checkpoint)

    # TensorRT checks
    results['TensorRT'] = check_tensorrt()
    if results['TensorRT'] and args.checkpoint:
        results['TRT Engines'] = check_trt_engines(args.trt_dir)

    # Quick test
    if results['CUDA'] and results['Model Imports']:
        results['Quick Test'] = run_quick_test()

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:<20} {status}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - System ready for inference!")
        print("\nNext steps:")
        print("1. Convert models to TRT (if not done):")
        print("   python -m inference_pipeline.tensorrt.convert_to_trt --checkpoint <path> --models all")
        print("\n2. Run tests:")
        print("   python -m inference_pipeline.tests.test_embedding_cache")
        print("   python -m inference_pipeline.tests.test_trt_conversion --checkpoint <path>")
        print("\n3. Start inference:")
        print("   python -m inference_pipeline.main --checkpoint <path> --use-trt")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues above before proceeding")
        print("\nCommon fixes:")
        print("- Install PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("- Install TensorRT: pip install tensorrt pycuda")
        print("- Add src/ to PYTHONPATH")
    print("="*60)


if __name__ == "__main__":
    main()
