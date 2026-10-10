"""Inference-only Qwen3 layer streaming with unchanged weights and KV caches.

Install on an unevaluated model loaded with lazy=True. Selected layers remain
resident; the others reload only their frozen base tensors for each forward.
All learned LoRA factors stay attached. This is deliberately not a training API.
"""
import json
import struct
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


class StreamedQwenLayer:
    def __init__(self, layer, index, files, boundary):
        if layer.training:
            raise ValueError('Streaming wrapper is inference-only')
        self.layer = layer
        self.index = index
        self.boundary = boundary
        self.mapping = {}
        self.expected = {}
        for name,value in tree_flatten(layer.parameters()):
            if name.endswith(('.lora_a','.lora_b')):
                continue
            source = f'model.layers.{index}.' + name.replace('.linear.','.')
            if source not in files:
                raise ValueError(f'No exact base tensor for {source}')
            self.mapping[name] = (files[source],source)
            self.expected[name] = (value.shape,value.dtype)
        self.release()

    def release(self):
        # Empty arrays remain valid update targets but allocate no weight storage.
        self.layer.update(tree_unflatten([(k,mx.zeros((0,),dtype=dtype))
                          for k,(_,dtype) in self.expected.items()]))

    def __call__(self, x, mask=None, cache=None):
        if self.layer.training:
            raise ValueError('Cannot train an inference streaming wrapper')
        needed = {}
        for target,(path,source) in self.mapping.items():
            needed.setdefault(path,{})[source] = target
        selected = {}
        for path,names in needed.items():
            lazy = mx.load(str(path))
            selected.update({target:lazy[source] for source,target in names.items()})
            del lazy
        for name,value in selected.items():
            if (value.shape,value.dtype) != self.expected[name]:
                raise ValueError(f'Base tensor metadata changed: {name}')
        self.layer.update(tree_unflatten(list(selected.items())))
        del selected
        try:
            output = self.layer(x,mask,cache)
            mx.eval(output, cache.state if cache is not None else [])
        finally:
            self.release()
        if self.index % 6 == 5:
            self.boundary(self.index)
        return output


def tensor_index(model_path):
    index = {}
    for path in sorted(Path(model_path).glob('model*.safetensors')):
        with path.open('rb') as stream:
            length = struct.unpack('<Q',stream.read(8))[0]
            if length > 16_000_000: raise ValueError('Invalid tensor header')
            header = json.loads(stream.read(length))
        for name in header:
            if name == '__metadata__': continue
            if name in index: raise ValueError('Duplicate base tensor')
            index[name] = path
    return index


def install_streaming(model, model_path, resident_layers=8, boundary=lambda index:None):
    if model.model_type != 'qwen3' or model.training:
        raise ValueError('Expected an evaluation-mode Qwen3 model')
    if not 0 <= resident_layers < len(model.layers):
        raise ValueError('Invalid resident layer count')
    index = tensor_index(model_path)
    layers = model.layers
    model.model.layers = [layer if i < resident_layers else
        StreamedQwenLayer(layer,i,index,boundary) for i,layer in enumerate(layers)]
    return model


def install_frozen_prefix(model, model_path, frozen_layers, boundary=lambda index:None):
    """Stream only a frozen eval-mode prefix while caching its forward features.

    This must run outside differentiation. Callers must separately retain the
    frozen adapter export factors, since wrappers hide unloaded parameters.
    """
    if model.model_type != 'qwen3' or not 0 < frozen_layers < len(model.layers):
        raise ValueError('Expected a proper frozen Qwen3 prefix')
    for layer in model.layers[:frozen_layers]:
        if layer.training or tree_flatten(layer.trainable_parameters()):
            raise ValueError('Every streamed prefix layer must be frozen and eval-mode')
    index = tensor_index(model_path)
    model.model.layers = [StreamedQwenLayer(layer,i,index,boundary) if i < frozen_layers else layer
                          for i,layer in enumerate(model.layers)]
    return model
