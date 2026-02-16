#!/bin/bash
# Quick test runner for inference pipeline optimizations
# This script runs all validation and tests in the correct order

echo "================================================================================"
echo "CS2 Inference Pipeline - Optimization Test Suite"
echo "================================================================================"
echo ""

# Set Python path
PYTHON="python3"

# Check if Python exists
if ! command -v $PYTHON &> /dev/null; then
    echo "ERROR: Python not found"
    echo "Please install Python 3.8+ with CUDA support"
    exit 1
fi

# Get checkpoint path from argument
CHECKPOINT="$1"
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint path required"
    echo "Usage: ./run_optimization_tests.sh path/to/checkpoint.pth"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
echo ""

# ============================================================
echo "STEP 1: Validating setup..."
echo "============================================================"
$PYTHON -m inference_pipeline.tests.validate_setup --checkpoint "$CHECKPOINT" --trt-dir ./trt_engines
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Setup validation failed!"
    echo "Please fix the issues above before continuing."
    exit 1
fi
echo ""
echo "✅ Setup validation passed"
echo ""
read -p "Press Enter to continue..."

# ============================================================
echo "STEP 2: Testing embedding cache..."
echo "============================================================"
$PYTHON -m inference_pipeline.tests.test_embedding_cache
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Embedding cache tests failed!"
    exit 1
fi
echo ""
echo "✅ Embedding cache tests passed"
echo ""
read -p "Press Enter to continue..."

# ============================================================
echo "STEP 3: Checking TRT engines..."
echo "============================================================"
if [ ! -f "trt_engines/radar_encoder.trt" ]; then
    echo ""
    echo "TRT engines not found. Converting models..."
    echo "This will take 5-10 minutes..."
    echo ""
    $PYTHON -m inference_pipeline.tensorrt.convert_to_trt --checkpoint "$CHECKPOINT" --output-dir ./trt_engines --models all --workspace-gb 4
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: TRT conversion failed!"
        echo "You can still use cache-only optimization (5x speedup)"
        echo "Or fix the TRT issues and try again"
        read -p "Press Enter to continue..."
        skip_trt=true
    else
        echo ""
        echo "✅ TRT conversion complete"
    fi
else
    echo "✅ TRT engines found (skipping conversion)"
fi
echo ""
read -p "Press Enter to continue..."

if [ "$skip_trt" != "true" ]; then
    # ============================================================
    echo "STEP 4: Testing TRT conversion..."
    echo "============================================================"
    $PYTHON -m inference_pipeline.tests.test_trt_conversion --checkpoint "$CHECKPOINT" --trt-dir ./trt_engines --models all
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: TRT conversion tests failed!"
        echo "You can still use cache-only optimization (5x speedup)"
        read -p "Press Enter to continue..."
        skip_trt=true
    else
        echo ""
        echo "✅ TRT conversion tests passed"
    fi
    echo ""
    read -p "Press Enter to continue..."

    # ============================================================
    echo "STEP 5: Benchmarking full pipeline..."
    echo "============================================================"
    echo "This will compare all optimization configurations (takes ~5 minutes)"
    echo ""
    $PYTHON -m inference_pipeline.tests.test_full_pipeline --checkpoint "$CHECKPOINT" --trt-dir ./trt_engines --compare-all
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Full pipeline tests failed!"
        exit 1
    fi
    echo ""
    echo "✅ Full pipeline tests passed"
    echo ""
fi

if [ "$skip_trt" = "true" ]; then
    echo ""
    echo "Skipping TRT tests (use cache-only optimization)"
    echo "You can still achieve 5x speedup with embedding cache alone!"
    echo ""
fi

echo "================================================================================"
echo "ALL TESTS COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. Run inference with optimizations:"
echo "   $PYTHON -m inference_pipeline.main --checkpoint \"$CHECKPOINT\" --use-trt --trt-dir ./trt_engines"
echo ""
echo "2. See documentation:"
echo "   - inference_pipeline/QUICKSTART_OPTIMIZATION.md"
echo "   - inference_pipeline/tensorrt/README.md"
echo "   - IMPLEMENTATION_SUMMARY.md"
echo ""
