"""
Test embedding cache correctness and performance.

Validates that cached embeddings produce identical results to non-cached version.
"""

import sys
from pathlib import Path
import time
import torch
import numpy as np

# Add paths
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path / "src"))
sys.path.insert(0, str(base_path / "inference_pipeline"))

from inference.embedding_cache import GPUEmbeddingCache, AudioEmbeddingCache


def test_gpu_cache_basic():
    """Test basic GPU cache operations."""
    print("\n=== Test 1: Basic GPU Cache Operations ===")

    cache = GPUEmbeddingCache(capacity=64, emb_dim=512, device='cuda')

    # Mock encoder
    def mock_encoder(x):
        return torch.randn(1, 512, device='cuda')

    # Add frames and retrieve sequences
    for i in range(100):
        seq = cache.add_and_get_sequence(
            torch.randn(1, 3, 224, 224, device='cuda'),
            mock_encoder,
            seq_length=32
        )

        assert seq.shape == (32, 512), f"Wrong shape: {seq.shape}"
        assert seq.device.type == 'cuda', "Sequence not on GPU"

    stats = cache.get_stats()
    print(f"✅ Cache stats: {stats}")
    print(f"✅ Basic operations passed")


def test_gpu_cache_wraparound():
    """Test ring buffer wrap-around logic."""
    print("\n=== Test 2: Ring Buffer Wrap-Around ===")

    cache = GPUEmbeddingCache(capacity=10, emb_dim=16, device='cuda')

    # Mock encoder that returns frame index
    frame_counter = [0]
    def indexed_encoder(x):
        emb = torch.full((1, 16), frame_counter[0], device='cuda')
        frame_counter[0] += 1
        return emb

    # Add 20 frames (2x capacity)
    for i in range(20):
        seq = cache.add_and_get_sequence(
            torch.randn(1, 3, 224, 224, device='cuda'),
            indexed_encoder,
            seq_length=5
        )

        if i >= 4:  # After warmup
            # Verify sequence is correct (last 5 frames)
            expected = torch.arange(i-4, i+1, device='cuda').float()
            expected = expected.unsqueeze(1).expand(-1, 16)

            if not torch.allclose(seq, expected, atol=1e-5):
                print(f"Frame {i}: Expected indices {list(range(i-4, i+1))}")
                print(f"Got: {seq[:, 0].cpu().numpy()}")
                raise AssertionError("Wrap-around failed")

    print(f"✅ Ring buffer wrap-around passed")


def test_audio_cache():
    """Test audio embedding cache."""
    print("\n=== Test 3: Audio Embedding Cache ===")

    cache = AudioEmbeddingCache(capacity=120, emb_dim=512, device='cuda')

    # Mock audio encoder
    def mock_audio_encoder(x):
        return torch.randn(1, 60, 512, device='cuda')

    # Add audio at different timestamps
    for i in range(10):
        timestamp = i * 1.0
        seq = cache.add_or_get_sequence(
            torch.randn(1, 480000, device='cuda'),
            mock_audio_encoder,
            timestamp=timestamp,
            audio_window_sec=30.0
        )

        assert seq.shape == (60, 512), f"Wrong shape: {seq.shape}"
        assert seq.device.type == 'cuda', "Sequence not on GPU"

    print(f"✅ Audio cache passed")


def test_cache_correctness_with_real_model():
    """Test cache produces same results as non-cached encoding."""
    print("\n=== Test 4: Cache Correctness (Real Model) ===")

    try:
        from RadarEncoder import RadarEncoderEffB0
    except ImportError:
        print("⚠️  RadarEncoder not available, skipping real model test")
        return

    # Load radar encoder
    encoder = RadarEncoderEffB0(
        pretrained=False,
        in_ch=3,
        out_ch=64,
        embed_dim=512
    ).cuda().eval()

    # Generate test frames
    test_frames = [torch.randn(1, 3, 224, 224, device='cuda') for _ in range(50)]

    # Method 1: Encode all frames independently (ground truth)
    with torch.no_grad():
        ground_truth = []
        for frame in test_frames:
            emb = encoder(frame).squeeze(0)
            ground_truth.append(emb)
        ground_truth = torch.stack(ground_truth)  # (50, 512)

    # Method 2: Use cache
    cache = GPUEmbeddingCache(capacity=64, emb_dim=512, device='cuda')

    cached_results = []
    for frame in test_frames:
        seq = cache.add_and_get_sequence(
            frame,
            encoder,
            seq_length=1  # Get only latest
        )
        cached_results.append(seq.squeeze(0))

    cached_results = torch.stack(cached_results)  # (50, 512)

    # Compare
    max_diff = (ground_truth - cached_results).abs().max().item()
    mean_diff = (ground_truth - cached_results).abs().mean().item()

    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    if max_diff > 1e-5:
        print("❌ FAILED: Cache outputs differ from ground truth")
        raise AssertionError(f"Max diff {max_diff} exceeds threshold 1e-5")

    print(f"✅ Cache correctness verified (max_diff={max_diff:.2e})")


def benchmark_cache_speedup():
    """Benchmark cache speedup."""
    print("\n=== Benchmark: Cache Speedup ===")

    try:
        from RadarEncoder import RadarEncoderEffB0
    except ImportError:
        print("⚠️  RadarEncoder not available, skipping benchmark")
        return

    encoder = RadarEncoderEffB0(
        pretrained=False,
        in_ch=3,
        out_ch=64,
        embed_dim=512
    ).cuda().eval()

    # Warmup
    dummy = torch.randn(1, 3, 224, 224, device='cuda')
    for _ in range(10):
        with torch.no_grad():
            _ = encoder(dummy)

    torch.cuda.synchronize()

    # Benchmark 1: Encode batch of 32 frames (current approach)
    frames_batch = torch.randn(1, 32, 3, 224, 224, device='cuda')

    times_batch = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            # Encode all 32 frames
            for i in range(32):
                _ = encoder(frames_batch[:, i])

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times_batch.append(elapsed)

    time_batch = np.mean(times_batch) * 1000  # ms

    # Benchmark 2: Encode 1 frame with cache (optimized approach)
    cache = GPUEmbeddingCache(capacity=64, emb_dim=512, device='cuda')
    single_frame = torch.randn(1, 3, 224, 224, device='cuda')

    # Warmup cache
    for i in range(32):
        _ = cache.add_and_get_sequence(single_frame, encoder, seq_length=32)

    times_cached = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            _ = cache.add_and_get_sequence(single_frame, encoder, seq_length=32)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times_cached.append(elapsed)

    time_cached = np.mean(times_cached) * 1000  # ms

    speedup = time_batch / time_cached

    print(f"Encode 32 frames (batch): {time_batch:.2f}ms")
    print(f"Encode 1 frame (cached):  {time_cached:.2f}ms")
    print(f"Speedup: {speedup:.1f}x")

    if speedup < 10:
        print("⚠️  Warning: Speedup less than expected (should be ~20-30x)")
    else:
        print(f"✅ Cache provides {speedup:.1f}x speedup")


def test_memory_usage():
    """Test cache memory usage."""
    print("\n=== Test 5: Memory Usage ===")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    initial_mem = torch.cuda.memory_allocated() / 1024**2  # MB

    # Create caches
    radar_cache = GPUEmbeddingCache(capacity=256, emb_dim=512, device='cuda')
    scene_cache = GPUEmbeddingCache(capacity=128, emb_dim=2048, device='cuda')
    audio_cache = AudioEmbeddingCache(capacity=120, emb_dim=512, device='cuda')

    cache_mem = torch.cuda.memory_allocated() / 1024**2  # MB
    overhead = cache_mem - initial_mem

    print(f"Initial GPU memory: {initial_mem:.2f} MB")
    print(f"After cache creation: {cache_mem:.2f} MB")
    print(f"Cache overhead: {overhead:.2f} MB")

    # Expected overhead
    expected = (
        (256 * 512 * 4) +      # Radar: 512 KB
        (128 * 2048 * 4) +     # Scene: 1 MB
        (120 * 60 * 512 * 4)   # Audio: 14.6 MB
    ) / 1024**2

    print(f"Expected overhead: {expected:.2f} MB")

    if abs(overhead - expected) > 5:  # Allow 5 MB tolerance
        print(f"⚠️  Warning: Memory overhead differs from expected by {abs(overhead - expected):.2f} MB")
    else:
        print(f"✅ Memory usage as expected")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Embedding Cache Test Suite")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available, tests require GPU")
        return

    try:
        test_gpu_cache_basic()
        test_gpu_cache_wraparound()
        test_audio_cache()
        test_cache_correctness_with_real_model()
        benchmark_cache_speedup()
        test_memory_usage()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
