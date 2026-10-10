"""A compact virtual safetensor file over selected ranges of the original file.

MLX's file-object loader evaluates synchronously into its own buffers. Only the
selected tensors appear in this view, so it never constructs the whole model's
lazy tensor graph. No duplicate weight files or NumPy staging buffers are used.
"""
import bisect
import io
import json
import math
import struct
import mlx.core as mx
from moshi_weight_stream import TensorRanges


class RangeView(io.RawIOBase):
    def __init__(self, path, header, segments):
        super().__init__()
        self.source = open(path, 'rb', buffering=0)
        self.header = header
        self.segments = segments
        self.starts = [row[0] for row in segments]
        self.length = segments[-1][1] if segments else len(header)
        self.position = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset if whence == 2 else -1
        if position < 0: raise ValueError('Invalid file position')
        self.position = position
        return position

    def readinto(self, buffer):
        target = memoryview(buffer).cast('B')
        written = 0
        while written < len(target) and self.position < self.length:
            if self.position < len(self.header):
                count = min(len(target)-written, len(self.header)-self.position)
                target[written:written+count] = self.header[self.position:self.position+count]
            else:
                index = bisect.bisect_right(self.starts, self.position)-1
                begin, end, physical = self.segments[index]
                if not begin <= self.position < end: raise ValueError('Noncontiguous tensor view')
                count = min(len(target)-written, end-self.position)
                self.source.seek(physical+self.position-begin)
                count = self.source.readinto(target[written:written+count])
                if not count: raise EOFError('Truncated source tensor')
            self.position += count
            written += count
        return written

    def close(self):
        self.source.close()
        super().close()


class SelectiveNativeRanges(TensorRanges):
    def __init__(self, path):
        super().__init__(path)
        self.plans = {}

    def plan(self, prefixes, strip):
        key = (tuple(prefixes), strip)
        if key in self.plans: return self.plans[key]
        selected = [(k,v) for k,v in self.header.items()
                    if k != '__metadata__' and k.startswith(tuple(prefixes))]
        if not selected: raise ValueError('No selected tensors')
        selected.sort(key=lambda row: row[1]['data_offsets'][0])
        sizes = {'BF16':2, 'F32':4, 'U32':4}
        header = {}; segments = []; offset = 0
        for name, metadata in selected:
            if not name.startswith(strip): raise ValueError('Incorrect prefix remapping')
            begin,end = metadata['data_offsets']
            size = math.prod(metadata['shape']) * sizes[metadata['dtype']]
            if begin < 0 or end-begin != size or self.offset+end > self.path.stat().st_size:
                raise ValueError('Invalid source tensor range')
            header[name[len(strip):]] = dict(metadata, data_offsets=[offset,offset+size])
            segments.append((offset,offset+size,self.offset+begin)); offset += size
        raw = json.dumps(header,separators=(',',':')).encode()
        raw += b' '*(-len(raw)%8)
        raw = struct.pack('<Q',len(raw))+raw
        segments = [(begin+len(raw),end+len(raw),physical) for begin,end,physical in segments]
        self.plans[key] = (raw,segments)
        return raw,segments

    def load(self, prefixes, strip=''):
        header,segments = self.plan(prefixes,strip)
        with RangeView(self.path,header,segments) as view:
            arrays = mx.load(view,format='safetensors')
            mx.eval(arrays)
        return arrays
