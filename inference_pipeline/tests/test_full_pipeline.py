"""
Integration test for full inference pipeline with cache + TRT optimizations.

Tests the complete flow from frame capture to action output.
"""

import sys
from pathlib import Path
import time
import torch
import numpy as np
from typing import Optional

# Add paths
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path / "inference_pipeline"))


def benchmark_full_pipeline(
    checkpoint_path: str,
    use_cache: bool = True,
    use_trt: bool = False,
    use_audio: bool = False,
    trt_dir: str = "./trt_engines",
    num_iterations: int = 100
):
    """
    Benchmark full inference pipeline with different optimization configs.

    Args:
        checkpoint_path: Path to model checkpoint
        use_cache: Enable embedding cache
        use_trt: Enable TensorRT acceleration
        use_audio: Enable audio processing
        trt_dir: Directory containing TRT engines
        num_iterations: Number of inference iterations to run

    Returns:
        Dict with timing statistics
    """
    from models.model_loader import load_models
    from inference.preprocessor import Preprocessor
    from inference.embedding_cache import GPUEmbeddingCache, AudioEmbeddingCache
    from config import InferenceConfig

    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Cache: {use_cache}")
    print(f"  TensorRT: {use_trt}")
    print(f"  Audio: {use_audio}")
    print(f"{'='*60}")

    # Load models
    print("Loading models...")
    models = load_models(
        checkpoint_path=checkpoint_path,
        device='cuda',
        use_audio=use_audio,
        use_trt=use_trt,
        trt_dir=trt_dir if use_trt else None
    )
    print("✅ Models loaded")

    # Create config
    config = InferenceConfig()
    preprocessor = Preprocessor(device='cuda', config=config)

    # Create caches if enabled
    radar_cache = None
    scene_cache = None
    audio_cache = None

    if use_cache:
        print("Initializing caches...")
        radar_cache = GPUEmbeddingCache(capacity=256, emb_dim=512, device='cuda')
        scene_cache = GPUEmbeddingCache(capacity=128, emb_dim=2048, device='cuda')
        if use_audio:
            audio_cache = AudioEmbeddingCache(capacity=120, emb_dim=512, device='cuda')
        print("✅ Caches initialized")

    # Generate mock data
    print("Generating test data...")
    radar_frames = torch.randn(config.radar_buffer_size, 3, 224, 224, device='cuda')
    scene_frames = torch.randn(config.scene_buffer_size, 3, 480, 640, device='cuda')
    audio_buffer = torch.randn(480000, device='cuda') if use_audio else None
    action_history = torch.randn(config.actions_buffer_size, 22, device='cuda')
    state_vec = torch.randn(95, device='cuda')

    # Warmup
    print("Warming up...")
    for _ in range(10):
        _run_single_inference(
            models, preprocessor, config,
            radar_frames, scene_frames, audio_buffer, action_history, state_vec,
            radar_cache, scene_cache, audio_cache, use_cache
        )

    torch.cuda.synchronize()
    print("✅ Warmup complete")

    # Benchmark
    print(f"\nRunning {num_iterations} iterations...")

    times = {
        'radar_encode': [],
        'scene_encode': [],
        'audio_encode': [],
        'temporal': [],
        'flow': [],
        'total': []
    }

    for i in range(num_iterations):
        torch.cuda.synchronize()
        start_total = time.time()

        timing = _run_single_inference(
            models, preprocessor, config,
            radar_frames, scene_frames, audio_buffer, action_history, state_vec,
            radar_cache, scene_cache, audio_cache, use_cache
        )

        torch.cuda.synchronize()
        total_time = time.time() - start_total

        times['radar_encode'].append(timing['radar'])
        times['scene_encode'].append(timing['scene'])
        times['audio_encode'].append(timing['audio'])
        times['temporal'].append(timing['temporal'])
        times['flow'].append(timing['flow'])
        times['total'].append(total_time)

        if (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}/{num_iterations} - {total_time*1000:.2f}ms")

    # Compute statistics
    stats = {}
    for key, values in times.items():
        values_ms = [v * 1000 for v in values if v > 0]  # Convert to ms, skip zeros
        if values_ms:
            stats[key] = {
                'mean': np.mean(values_ms),
                'std': np.std(values_ms),
                'min': np.min(values_ms),
                'max': np.max(values_ms),
                'median': np.median(values_ms)
            }
        else:
            stats[key] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}

    return stats


def _run_single_inference(
    models, preprocessor, config,
    radar_frames, scene_frames, audio_buffer, action_history, state_vec,
    radar_cache, scene_cache, audio_cache, use_cache
):
    """Run single inference and return timings."""
    timing = {'radar': 0, 'scene': 0, 'audio': 0, 'temporal': 0, 'flow': 0}

    # Radar encoding
    torch.cuda.synchronize()
    start = time.time()

    if use_cache and radar_cache is not None:
        latest_radar = radar_frames[-1:].unsqueeze(0)  # (1, 1, 3, 224, 224)
        radar_seq = radar_cache.add_and_get_sequence(
            latest_radar.squeeze(1),
            models.radar_encoder,
            seq_length=config.radar_buffer_size
        )
        radar_embeds = radar_seq.unsqueeze(0)  # (1, seq, 512)
    else:
        radar_tensor = radar_frames.unsqueeze(0)  # (1, seq, 3, 224, 224)
        radar_embeds = preprocessor.encode_radar_sequence(
            models.radar_encoder,
            radar_tensor
        )

    torch.cuda.synchronize()
    timing['radar'] = time.time() - start

    # Scene encoding
    torch.cuda.synchronize()
    start = time.time()

    if use_cache and scene_cache is not None:
        latest_scene = scene_frames[-1:].unsqueeze(0)  # (1, 1, 3, H, W)
        scene_seq = scene_cache.add_and_get_sequence(
            latest_scene.squeeze(1),
            lambda x: models.yolo.embeds(x),
            seq_length=config.scene_buffer_size
        )
        scene_embeds = scene_seq.unsqueeze(0)  # (1, seq, 2048)

        # Get detections from latest frame only
        with torch.no_grad():
            yolo_output = models.yolo(latest_scene.squeeze(1))
            detections = preprocessor._process_yolo_detections(yolo_output)
    else:
        scene_tensor = scene_frames.unsqueeze(0)
        scene_embeds, detections = preprocessor.encode_scene_sequence(
            models.yolo,
            scene_tensor
        )

    torch.cuda.synchronize()
    timing['scene'] = time.time() - start

    # Audio encoding
    if models.audio_encoder is not None and audio_buffer is not None:
        torch.cuda.synchronize()
        start = time.time()

        if use_cache and audio_cache is not None:
            audio_seq = audio_cache.add_or_get_sequence(
                audio_buffer.unsqueeze(0),
                models.audio_encoder,
                timestamp=time.time(),
                audio_window_sec=30.0
            )
            audio_embeds = audio_seq.unsqueeze(0)  # (1, 60, 512)
        else:
            with torch.no_grad():
                audio_embeds = models.audio_encoder(audio_buffer.unsqueeze(0))

        torch.cuda.synchronize()
        timing['audio'] = time.time() - start
    else:
        audio_embeds = None

    # Temporal transformer
    torch.cuda.synchronize()
    start = time.time()

    action_seq = action_history.unsqueeze(0)  # (1, 16, 22)
    state = state_vec.unsqueeze(0)  # (1, 95)

    with torch.no_grad():
        mouse_embed, policy_keys, value = models.temporal_model(
            radar_embeds,
            scene_embeds,
            detections,
            action_seq,
            state,
            audio_embeds
        )

    torch.cuda.synchronize()
    timing['temporal'] = time.time() - start

    # Flow head (if not TRT)
    torch.cuda.synchronize()
    start = time.time()

    if hasattr(models.flow_head, 'trt_engine'):
        # TRT already includes flow head in temporal
        policy_mouse = mouse_embed
    else:
        with torch.no_grad():
            policy_mouse = models.flow_head.sample(mouse_embed, deterministic=True)

    torch.cuda.synchronize()
    timing['flow'] = time.time() - start

    return timing


def compare_configurations(checkpoint_path: str, trt_dir: str = "./trt_engines"):
    """
    Compare all optimization configurations:
    1. Baseline (no cache, no TRT)
    2. Cache only
    3. TRT only
    4. Cache + TRT (fully optimized)
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*80)

    configs = [
        {"name": "1. Baseline (PyTorch, no cache)", "cache": False, "trt": False},
        {"name": "2. Cache only", "cache": True, "trt": False},
        {"name": "3. TRT only", "cache": False, "trt": True},
        {"name": "4. Cache + TRT (fully optimized)", "cache": True, "trt": True},
    ]

    results = []

    for cfg in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {cfg['name']}")
        print(f"{'='*80}")

        try:
            stats = benchmark_full_pipeline(
                checkpoint_path=checkpoint_path,
                use_cache=cfg['cache'],
                use_trt=cfg['trt'],
                use_audio=False,  # Disable audio for consistent comparison
                trt_dir=trt_dir,
                num_iterations=50
            )

            results.append({
                'name': cfg['name'],
                'cache': cfg['cache'],
                'trt': cfg['trt'],
                'stats': stats
            })

        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'name': cfg['name'],
                'cache': cfg['cache'],
                'trt': cfg['trt'],
                'stats': None,
                'error': str(e)
            })

    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Configuration':<40} {'Total (ms)':<15} {'FPS':<10} {'Speedup':<10}")
    print("-"*80)

    baseline_time = None
    for r in results:
        if r['stats'] is not None:
            total_time = r['stats']['total']['mean']
            fps = 1000.0 / total_time if total_time > 0 else 0

            if baseline_time is None:
                baseline_time = total_time
                speedup = 1.0
            else:
                speedup = baseline_time / total_time if total_time > 0 else 0

            print(f"{r['name']:<40} {total_time:>8.2f}      {fps:>6.1f}     {speedup:>6.2f}x")
        else:
            print(f"{r['name']:<40} {'ERROR':<15} {'-':<10} {'-':<10}")

    # Detailed breakdown for fully optimized
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN (Fully Optimized: Cache + TRT)")
    print("="*80)

    fully_optimized = next((r for r in results if r['cache'] and r['trt']), None)
    if fully_optimized and fully_optimized['stats']:
        stats = fully_optimized['stats']
        print(f"{'Component':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'% of Total':<12}")
        print("-"*80)

        total = stats['total']['mean']
        for key in ['radar_encode', 'scene_encode', 'audio_encode', 'temporal', 'flow']:
            mean = stats[key]['mean']
            std = stats[key]['std']
            pct = (mean / total * 100) if total > 0 and mean > 0 else 0

            display_name = key.replace('_', ' ').title()
            print(f"{display_name:<20} {mean:>8.2f}     {std:>8.2f}     {pct:>8.1f}%")

        print("-"*80)
        print(f"{'Total':<20} {total:>8.2f}")
        print(f"{'FPS':<20} {1000.0/total:>8.1f}")

    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Test full inference pipeline")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--trt-dir", type=str, default="./trt_engines",
                        help="Directory containing TRT engines")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all optimization configurations")
    parser.add_argument("--use-cache", action="store_true",
                        help="Enable embedding cache")
    parser.add_argument("--use-trt", action="store_true",
                        help="Enable TensorRT")
    parser.add_argument("--use-audio", action="store_true",
                        help="Enable audio processing")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations for benchmark")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    if args.compare_all:
        compare_configurations(args.checkpoint, args.trt_dir)
    else:
        stats = benchmark_full_pipeline(
            checkpoint_path=args.checkpoint,
            use_cache=args.use_cache,
            use_trt=args.use_trt,
            use_audio=args.use_audio,
            trt_dir=args.trt_dir,
            num_iterations=args.iterations
        )

        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        total = stats['total']['mean']
        print(f"Total inference time: {total:.2f} ms")
        print(f"FPS: {1000.0/total:.1f}")
        print("\nBreakdown:")
        for key in ['radar_encode', 'scene_encode', 'audio_encode', 'temporal', 'flow']:
            mean = stats[key]['mean']
            if mean > 0:
                print(f"  {key}: {mean:.2f} ms ({mean/total*100:.1f}%)")


if __name__ == "__main__":
    main()
