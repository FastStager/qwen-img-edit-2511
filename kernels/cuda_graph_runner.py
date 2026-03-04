"""
CUDA Graph capture for transformer forward pass.

Captures the transformer as a CUDA graph for the standard 1024x1024 latent shape,
eliminating kernel launch overhead across denoising steps. Falls back to eager
execution for non-standard resolutions.
"""

import torch
from typing import Optional, Tuple


class CUDAGraphRunner:
    """
    Wraps a transformer module with CUDA graph replay for a fixed input shape.
    Uses static input/output buffers with copy_() before graph.replay().
    """

    def __init__(self, transformer, standard_latent_shape: Tuple[int, ...]):
        """
        transformer: the diffusion transformer module
        standard_latent_shape: e.g. (1, 16, 128, 128) for 1024x1024 at 8x compression
        """
        self.transformer = transformer
        self.standard_shape = standard_latent_shape
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_inputs = {}
        self.static_output = None
        self._captured = False

    def capture(self, sample_inputs: dict):
        """
        Capture the transformer forward pass as a CUDA graph.
        Must be called with torch.no_grad() (not inference_mode).

        sample_inputs: dict of keyword arguments to transformer.forward()
        """
        if self._captured:
            return

        device = next(self.transformer.parameters()).device

        # Warmup runs (required before capture)
        for _ in range(3):
            with torch.no_grad():
                _ = self.transformer(**sample_inputs)

        torch.cuda.synchronize()

        # Create static input buffers
        self.static_inputs = {}
        for key, val in sample_inputs.items():
            if isinstance(val, torch.Tensor):
                self.static_inputs[key] = val.clone().to(device)
            else:
                self.static_inputs[key] = val

        # Capture
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                self.static_output = self.transformer(**self.static_inputs)

        self._captured = True
        print(f"--- CUDA Graph captured for shape {self.standard_shape} ---")

    def forward(self, **kwargs) -> object:
        """
        Run transformer forward. Uses graph replay if input shapes match,
        otherwise falls back to eager execution.
        """
        # Check if shapes match the captured graph
        can_replay = self._captured
        if can_replay:
            for key, val in kwargs.items():
                if isinstance(val, torch.Tensor) and key in self.static_inputs:
                    if val.shape != self.static_inputs[key].shape:
                        can_replay = False
                        break

        if can_replay:
            # Copy inputs into static buffers
            for key, val in kwargs.items():
                if isinstance(val, torch.Tensor) and key in self.static_inputs:
                    self.static_inputs[key].copy_(val)
                elif key in self.static_inputs:
                    self.static_inputs[key] = val

            # Replay
            self.graph.replay()
            return self.static_output
        else:
            # Fallback: eager execution for non-standard shapes
            with torch.no_grad():
                return self.transformer(**kwargs)

    @property
    def is_captured(self):
        return self._captured


def setup_cuda_graph(pipe, height=1024, width=1024):
    """
    Set up CUDA graph capture for the transformer in the pipeline.
    Returns a CUDAGraphRunner instance, or None if capture fails.

    Must be called AFTER model is loaded and quantized, but BEFORE serving.
    """
    try:
        # Compute latent shape: 8x spatial compression, 16 channels for Qwen
        latent_h = height // 8
        latent_w = width // 8
        latent_channels = pipe.transformer.config.in_channels if hasattr(pipe.transformer, 'config') else 16
        latent_shape = (1, latent_channels, latent_h, latent_w)

        runner = CUDAGraphRunner(pipe.transformer, latent_shape)

        # Build sample inputs for capture
        device = next(pipe.transformer.parameters()).device
        dtype = next(pipe.transformer.parameters()).dtype

        # The exact input signature depends on the transformer architecture.
        # For QwenImageEdit, we need to probe the forward method signature.
        import inspect
        sig = inspect.signature(pipe.transformer.forward)
        param_names = list(sig.parameters.keys())

        print(f"--- CUDA Graph: transformer forward params: {param_names} ---")

        # Create dummy inputs matching the transformer's expected signature
        sample_inputs = {}

        # Common inputs for DiT-style transformers
        if 'hidden_states' in param_names:
            sample_inputs['hidden_states'] = torch.randn(latent_shape, device=device, dtype=dtype)
        if 'timestep' in param_names:
            sample_inputs['timestep'] = torch.tensor([500.0], device=device, dtype=dtype)
        if 'encoder_hidden_states' in param_names:
            # Text encoder output — typical sequence length for short prompts
            sample_inputs['encoder_hidden_states'] = torch.randn(1, 77, 4096, device=device, dtype=dtype)
        if 'return_dict' in param_names:
            sample_inputs['return_dict'] = False

        # Attempt capture with torch.no_grad (not inference_mode, which blocks graph capture)
        with torch.no_grad():
            runner.capture(sample_inputs)

        return runner

    except Exception as e:
        print(f"--- CUDA Graph capture failed (non-fatal): {e} ---")
        return None
