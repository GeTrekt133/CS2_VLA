@echo off
REM Quick test runner for inference pipeline optimizations
REM This script runs all validation and tests in the correct order

echo ================================================================================
echo CS2 Inference Pipeline - Optimization Test Suite
echo ================================================================================
echo.

REM Set Python path
set PYTHON=C:/Users/misas/AppData/Local/Programs/Python/Python313/python.exe

REM Check if Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo Please update the PYTHON variable in this script
    exit /b 1
)

REM Get checkpoint path from argument or use default
set CHECKPOINT=%1
if "%CHECKPOINT%"=="" (
    echo ERROR: Checkpoint path required
    echo Usage: run_optimization_tests.bat path\to\checkpoint.pth
    exit /b 1
)

if not exist "%CHECKPOINT%" (
    echo ERROR: Checkpoint not found: %CHECKPOINT%
    exit /b 1
)

echo Using checkpoint: %CHECKPOINT%
echo.

REM ============================================================
echo STEP 1: Validating setup...
echo ============================================================
%PYTHON% -m inference_pipeline.tests.validate_setup --checkpoint "%CHECKPOINT%" --trt-dir ./trt_engines
if errorlevel 1 (
    echo.
    echo ERROR: Setup validation failed!
    echo Please fix the issues above before continuing.
    exit /b 1
)
echo.
echo ✅ Setup validation passed
echo.
pause

REM ============================================================
echo STEP 2: Testing embedding cache...
echo ============================================================
%PYTHON% -m inference_pipeline.tests.test_embedding_cache
if errorlevel 1 (
    echo.
    echo ERROR: Embedding cache tests failed!
    exit /b 1
)
echo.
echo ✅ Embedding cache tests passed
echo.
pause

REM ============================================================
echo STEP 3: Checking TRT engines...
echo ============================================================
if not exist "trt_engines\radar_encoder.trt" (
    echo.
    echo TRT engines not found. Converting models...
    echo This will take 5-10 minutes...
    echo.
    %PYTHON% -m inference_pipeline.tensorrt.convert_to_trt --checkpoint "%CHECKPOINT%" --output-dir ./trt_engines --models all --workspace-gb 4
    if errorlevel 1 (
        echo.
        echo ERROR: TRT conversion failed!
        echo You can still use cache-only optimization (5x speedup)
        echo Or fix the TRT issues and try again
        pause
        goto skip_trt
    )
    echo.
    echo ✅ TRT conversion complete
) else (
    echo ✅ TRT engines found (skipping conversion)
)
echo.
pause

REM ============================================================
echo STEP 4: Testing TRT conversion...
echo ============================================================
%PYTHON% -m inference_pipeline.tests.test_trt_conversion --checkpoint "%CHECKPOINT%" --trt-dir ./trt_engines --models all
if errorlevel 1 (
    echo.
    echo ERROR: TRT conversion tests failed!
    echo You can still use cache-only optimization (5x speedup)
    pause
    goto skip_trt
)
echo.
echo ✅ TRT conversion tests passed
echo.
pause

REM ============================================================
echo STEP 5: Benchmarking full pipeline...
echo ============================================================
echo This will compare all optimization configurations (takes ~5 minutes)
echo.
%PYTHON% -m inference_pipeline.tests.test_full_pipeline --checkpoint "%CHECKPOINT%" --trt-dir ./trt_engines --compare-all
if errorlevel 1 (
    echo.
    echo ERROR: Full pipeline tests failed!
    exit /b 1
)
echo.
echo ✅ Full pipeline tests passed
echo.

goto end

:skip_trt
echo.
echo Skipping TRT tests (use cache-only optimization)
echo You can still achieve 5x speedup with embedding cache alone!
echo.

:end
echo ================================================================================
echo ALL TESTS COMPLETE!
echo ================================================================================
echo.
echo Next steps:
echo 1. Run inference with optimizations:
echo    %PYTHON% -m inference_pipeline.main --checkpoint "%CHECKPOINT%" --use-trt --trt-dir ./trt_engines
echo.
echo 2. See documentation:
echo    - inference_pipeline\QUICKSTART_OPTIMIZATION.md
echo    - inference_pipeline\tensorrt\README.md
echo    - IMPLEMENTATION_SUMMARY.md
echo.
pause
