"""
TensorRT inference wrappers.

Provides PyTorch-like interface for TensorRT engines.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # Initializes CUDA driver
    HAS_TRT = True
except ImportError:
    HAS_TRT = False


class TRTEngine:
    """
    TensorRT engine wrapper with PyTorch-like interface.

    Usage:
        engine = TRTEngine("radar_encoder.trt")
        output = engine(input_tensor)  # PyTorch tensor in, PyTorch tensor out
    """

    def __init__(self, engine_path: str, device: str = 'cuda'):
        """
        Load TensorRT engine from file.

        Args:
            engine_path: Path to .trt engine file
            device: CUDA device (currently only 'cuda' supported)
        """
        if not HAS_TRT:
            raise RuntimeError(
                "TensorRT not installed. Install with:\n"
                "  pip install tensorrt pycuda"
            )

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get input/output names and shapes
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = tuple(shape)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(shape)

        # Preallocate device buffers for single input/output
        # (assumes single input, single output - typical for our encoders)
        assert len(self.input_names) == 1, "Multiple inputs not supported"
        assert len(self.output_names) == 1, "Multiple outputs not supported"

        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]

        input_shape = self.input_shapes[self.input_name]
        output_shape = self.output_shapes[self.output_name]

        # Allocate buffers
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize

        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)

        # Cache numpy output buffer
        self.output_buffer = np.empty(output_shape, dtype=np.float32)

        print(f"[TRT] Loaded engine: {self.engine_path.name}")
        print(f"  Input:  {self.input_name} {input_shape}")
        print(f"  Output: {self.output_name} {output_shape}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on input tensor.

        Args:
            x: Input tensor (must match engine input shape)

        Returns:
            Output tensor
        """
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy().astype(np.float32)
            return_torch = True
        else:
            x_np = x.astype(np.float32)
            return_torch = False

        # Verify shape
        expected_shape = self.input_shapes[self.input_name]
        if x_np.shape != expected_shape:
            raise ValueError(
                f"Input shape mismatch: got {x_np.shape}, expected {expected_shape}"
            )

        # Copy input to device
        cuda.memcpy_htod(self.d_input, x_np)

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        # Execute
        self.context.execute_async_v3(stream_handle=0)

        # Copy output back
        cuda.memcpy_dtoh(self.output_buffer, self.d_output)

        # Convert back to torch if needed
        if return_torch:
            return torch.from_numpy(self.output_buffer.copy()).to(self.device)
        else:
            return self.output_buffer.copy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for __call__ to match PyTorch API."""
        return self(x)

    def eval(self):
        """No-op for compatibility with PyTorch models."""
        return self

    def to(self, device):
        """No-op for compatibility with PyTorch models."""
        return self

    def __repr__(self):
        return f"TRTEngine({self.engine_path.name})"


class TRTRadarEncoder:
    """
    TensorRT wrapper for RadarEncoder.

    Drop-in replacement for PyTorch RadarEncoder.
    """

    def __init__(self, engine_path: str, device: str = 'cuda'):
        self.engine = TRTEngine(engine_path, device)
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 3, 224, 224) input tensor

        Returns:
            (B, 512) embeddings
        """
        return self.engine(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self


class TRTYOLOEmbed:
    """
    TensorRT wrapper for YOLO embedding extraction.

    Note: This only returns embeddings, not detections.
    For detections, use the full PyTorch YOLO model.
    """

    def __init__(self, engine_path: str, device: str = 'cuda'):
        self.engine = TRTEngine(engine_path, device)
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings only.

        Args:
            x: (B, 3, H, W) input tensor

        Returns:
            (B, 2048) embeddings
        """
        return self.engine(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self


class TRTAudioEncoder:
    """
    TensorRT wrapper for AudioEncoder.

    Drop-in replacement for PyTorch AudioEncoder.
    """

    def __init__(self, engine_path: str, device: str = 'cuda'):
        self.engine = TRTEngine(engine_path, device)
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 480000) audio waveform

        Returns:
            (B, 60, 512) audio embeddings
        """
        # TRT engine outputs (B, 60, 512) directly
        return self.engine(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self


class TRTTemporalFlow:
    """
    TensorRT wrapper for TemporalTransformer + FlowActionHead.

    Combines both models into single TRT engine for maximum optimization.
    Handles multi-input, multi-output inference.
    """

    def __init__(self, engine_path: str, device: str = 'cuda', use_audio: bool = True):
        if not HAS_TRT:
            raise RuntimeError(
                "TensorRT not installed. Install with:\n"
                "  pip install tensorrt pycuda"
            )

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self.device = device
        self.use_audio = use_audio
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get input/output tensor info
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = tuple(shape)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(shape)

        # Allocate GPU buffers
        self.d_inputs = {}
        self.d_outputs = {}

        for name, shape in self.input_shapes.items():
            size = np.prod(shape) * np.dtype(np.float32).itemsize
            self.d_inputs[name] = cuda.mem_alloc(size)

        for name, shape in self.output_shapes.items():
            size = np.prod(shape) * np.dtype(np.float32).itemsize
            self.d_outputs[name] = cuda.mem_alloc(size)

        # Cache output buffers
        self.output_buffers = {
            name: np.empty(shape, dtype=np.float32)
            for name, shape in self.output_shapes.items()
        }

        print(f"[TRT] Loaded TemporalFlow engine: {self.engine_path.name}")
        print(f"  Inputs:")
        for name, shape in self.input_shapes.items():
            print(f"    {name}: {shape}")
        print(f"  Outputs:")
        for name, shape in self.output_shapes.items():
            print(f"    {name}: {shape}")

    def __call__(
        self,
        radar_seq: torch.Tensor,
        scene_seq: torch.Tensor,
        detection_seq: torch.Tensor,
        action_seq: torch.Tensor,
        state_vec: torch.Tensor,
        audio_seq: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through TRT engine.

        Args:
            radar_seq: (B, T_radar, 512)
            scene_seq: (B, T_scene, 2048)
            detection_seq: (B, 1, 100)
            action_seq: (B, 16, 22)
            state_vec: (B, 95)
            audio_seq: (B, 60, 512) or None

        Returns:
            policy_mouse: (B, 2)
            policy_keys: (B, 20)
            value: (B, 1)
        """
        # Prepare inputs
        inputs = {
            'radar_seq': radar_seq.cpu().numpy().astype(np.float32),
            'scene_seq': scene_seq.cpu().numpy().astype(np.float32),
            'detection_seq': detection_seq.cpu().numpy().astype(np.float32),
            'action_seq': action_seq.cpu().numpy().astype(np.float32),
            'state_vec': state_vec.cpu().numpy().astype(np.float32),
        }

        if self.use_audio and audio_seq is not None:
            inputs['audio_seq'] = audio_seq.cpu().numpy().astype(np.float32)

        # Copy inputs to device
        for name, data in inputs.items():
            if name in self.d_inputs:
                cuda.memcpy_htod(self.d_inputs[name], data)

        # Set tensor addresses
        for name in self.input_names:
            if name in self.d_inputs:
                self.context.set_tensor_address(name, int(self.d_inputs[name]))

        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.d_outputs[name]))

        # Execute
        self.context.execute_async_v3(stream_handle=0)

        # Copy outputs back
        for name in self.output_names:
            cuda.memcpy_dtoh(self.output_buffers[name], self.d_outputs[name])

        # Convert to torch tensors
        policy_mouse = torch.from_numpy(self.output_buffers['policy_mouse'].copy()).to(self.device)
        policy_keys = torch.from_numpy(self.output_buffers['policy_keys'].copy()).to(self.device)
        value = torch.from_numpy(self.output_buffers['value'].copy()).to(self.device)

        return policy_mouse, policy_keys, value

    def forward(self, *args, **kwargs):
        """Alias for __call__."""
        return self(*args, **kwargs)

    def eval(self):
        """No-op for compatibility."""
        return self

    def to(self, device):
        """No-op for compatibility."""
        self.device = device
        return self

    def __repr__(self):
        return f"TRTTemporalFlow({self.engine_path.name}, use_audio={self.use_audio})"


def load_trt_models(trt_dir: str, device: str = 'cuda', use_audio: bool = True) -> dict:
    """
    Load all available TRT engines from directory.

    Args:
        trt_dir: Directory containing .trt files
        device: CUDA device
        use_audio: Whether audio is used (for temporal_flow loading)

    Returns:
        Dict with loaded TRT engines:
        {
            'radar_encoder': TRTRadarEncoder or None,
            'yolo_embed': TRTYOLOEmbed or None,
            'audio_encoder': TRTAudioEncoder or None,
            'temporal_flow': TRTTemporalFlow or None,
        }
    """
    trt_dir = Path(trt_dir)

    models = {
        'radar_encoder': None,
        'yolo_embed': None,
        'audio_encoder': None,
        'temporal_flow': None,
    }

    # Load radar encoder
    radar_path = trt_dir / "radar_encoder.trt"
    if radar_path.exists():
        try:
            models['radar_encoder'] = TRTRadarEncoder(str(radar_path), device)
            print(f"[TRT] Loaded RadarEncoder from {radar_path}")
        except Exception as e:
            print(f"[TRT] Failed to load RadarEncoder: {e}")

    # Load YOLO embed
    yolo_path = trt_dir / "yolo_embed.trt"
    if yolo_path.exists():
        try:
            models['yolo_embed'] = TRTYOLOEmbed(str(yolo_path), device)
            print(f"[TRT] Loaded YOLOEmbed from {yolo_path}")
        except Exception as e:
            print(f"[TRT] Failed to load YOLOEmbed: {e}")

    # Load audio encoder
    audio_path = trt_dir / "audio_encoder.trt"
    if audio_path.exists():
        try:
            models['audio_encoder'] = TRTAudioEncoder(str(audio_path), device)
            print(f"[TRT] Loaded AudioEncoder from {audio_path}")
        except Exception as e:
            print(f"[TRT] Failed to load AudioEncoder: {e}")

    # Load temporal transformer + flow head
    temporal_path = trt_dir / "temporal_flow.trt"
    if temporal_path.exists():
        try:
            models['temporal_flow'] = TRTTemporalFlow(str(temporal_path), device, use_audio=use_audio)
            print(f"[TRT] Loaded TemporalFlow from {temporal_path}")
        except Exception as e:
            print(f"[TRT] Failed to load TemporalFlow: {e}")

    return models


def benchmark_trt_vs_pytorch(pytorch_model, trt_engine, input_shape, num_iterations=100):
    """
    Benchmark TRT vs PyTorch model.

    Args:
        pytorch_model: PyTorch model
        trt_engine: TRT engine wrapper
        input_shape: Input tensor shape
        num_iterations: Number of iterations for benchmarking
    """
    import time

    device = 'cuda'
    pytorch_model.eval().to(device)

    # Warmup
    dummy_input = torch.randn(*input_shape, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_model(dummy_input)
        _ = trt_engine(dummy_input)

    torch.cuda.synchronize()

    # Benchmark PyTorch
    pytorch_times = []
    for _ in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            _ = pytorch_model(dummy_input)
        torch.cuda.synchronize()
        pytorch_times.append(time.time() - start)

    # Benchmark TRT
    trt_times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = trt_engine(dummy_input)
        torch.cuda.synchronize()
        trt_times.append(time.time() - start)

    pytorch_avg = np.mean(pytorch_times) * 1000  # ms
    trt_avg = np.mean(trt_times) * 1000  # ms
    speedup = pytorch_avg / trt_avg

    print(f"\n{'='*60}")
    print(f"Benchmark Results ({num_iterations} iterations)")
    print(f"{'='*60}")
    print(f"PyTorch: {pytorch_avg:.2f} ms")
    print(f"TensorRT: {trt_avg:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'='*60}\n")

    return speedup
