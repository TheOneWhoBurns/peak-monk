"""Use MLX's lazy safetensor Load nodes; select before evaluating any tensor.

Verified against MLX v0.26.5 mlx/io/safetensors.cpp: the header creates lazy
CPU-stream Load primitives. Discard unselected nodes before mx.eval.
"""
import mlx.core as mx
from moshi_weight_stream import TensorRanges
class NativeTensorRanges(TensorRanges):
    def load(self,prefixes,strip=''):
        lazy=mx.load(str(self.path))
        selected={key[len(strip):]:value for key,value in lazy.items() if key.startswith(tuple(prefixes))}
        if not selected or any(not k.startswith(strip) for k in lazy if k.startswith(tuple(prefixes))):raise ValueError('Invalid selection/remapping')
        del lazy
        mx.eval(selected)
        return selected
