"""Native duplex inference with selective assembly, no wiring, and resource guards.

The full quantized model is resident during generation. This is a separate,
explicitly measured inference experiment; the sub-600MB training measurement
must not be presented as its inference memory requirement.
"""
import argparse,gc,json,time,resource,subprocess
from dataclasses import asdict
from pathlib import Path
from bounded_speech_runtime import BoundedSpeechRuntime
from bounded_text_runtime import atomic_json,swap_bytes
from sample_dialogue_paths import sha

class InferenceRuntime(BoundedSpeechRuntime):
    def settle_load(self):
        """Allow OS reclamation at a synchronized load boundary, without work."""
        self.mx.synchronize();self.original_clear();deadline=time.monotonic()+10
        while True:
            pressure=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
            if (pressure!=2 or time.monotonic()>=deadline
                    or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4_600_000_000
                    or swap_bytes()-self.initial_swap>512*1024**2
                    or time.monotonic()-self.started>self.max_seconds):
                return self.response_done()
            self.event('load_settle_wait',memory_pressure_level=pressure,mlx_active_bytes=self.mx.get_active_memory())
            time.sleep(1)
    def response_done(self):
        self.mx.synchronize();self.original_clear()
        metrics=self.check()
        if metrics['mlx_peak_bytes']>self.peak_limit:
            self.event('resource_guard_stop',reason='native inference peak exceeds declared mode limit',**metrics)
            raise RuntimeError('Native inference peak exceeded declared allocation plan')
        return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--adapter');p.add_argument('--adapter-config');p.add_argument('--output',type=Path,required=True);p.add_argument('--stream-temporal',action='store_true');p.add_argument('--seed',type=int,default=41);p.add_argument('--max-frames',type=int);p.add_argument('--sampling-preset',choices=['local','offline'],default='local');a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier native evidence')
    if bool(a.adapter)!=bool(a.adapter_config):raise ValueError('Adapter and training config must be supplied together')
    cfg=json.loads(Path(a.adapter_config).read_text()) if a.adapter else dict(rank=8,scale=2.)
    if cfg.get('layers',32)!=32:raise ValueError('Selective assembly expects all32layer adapters')
    a.output.mkdir(parents=True)
    weight='models/moshi/model.q4.safetensors'
    assert sha(weight)=='7959d590e23c1ebc78cfa3501344a6ff331561aa0cadc4429b733b890bbc919c'
    atomic_json(a.output/'config.json',dict(input=str(a.input),input_sha256=sha(a.input),adapter=a.adapter,adapter_sha256=sha(a.adapter) if a.adapter else None,adapter_config=a.adapter_config,seed=a.seed,explicit_frame_limit=a.max_frames,stream_temporal=a.stream_temporal,sampling_preset=a.sampling_preset,source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['infer_moshi_unwired.py','assemble_moshi_parts.py','native_tensor_reader.py','moshi_weight_stream.py','bounded_speech_runtime.py','bounded_text_runtime.py']}))
    with InferenceRuntime(a.output,memory_limit=1_400_000_000 if a.stream_temporal else 5_100_000_000,max_seconds=1800) as runtime:
        runtime.peak_limit=1_500_000_000 if a.stream_temporal else 5_250_000_000
        import numpy as np
        import mlx.core as mx
        import sentencepiece as spm
        from moshi_mlx.models import LmGen,config_v0_1
        from moshi_mlx.utils import Sampler
        from moshi_weight_stream import MoshiParts
        from assemble_moshi_parts import assemble,assemble_streamed
        codes=np.load(a.input)['user_codes'];original_frames=codes.shape[1]
        assert codes.ndim==2 and codes.shape[0]==8 and np.all((codes>=0)&(codes<2048))
        if a.max_frames is not None:
            if a.max_frames<3:raise ValueError('Need at least3frames')
            codes=codes[:,:a.max_frames]
        mx.random.seed(a.seed);start=time.monotonic()
        parts=MoshiParts(weight,config_v0_1(),a.adapter,rank=cfg['rank'],scale=cfg['scale'])
        def boundary(phase,**details):
            metrics=runtime.settle_load() if phase.startswith('inference_') else runtime.response_done()
            runtime.event(phase,**details,**metrics)
        from native_tensor_reader import NativeTensorRanges
        parts.weights=NativeTensorRanges(weight)
        if a.adapter:parts.adapter=NativeTensorRanges(a.adapter)
        model=(assemble_streamed if a.stream_temporal else assemble)(parts,boundary=boundary);load_seconds=time.monotonic()-start
        start=time.monotonic();model.warmup();boundary('warmup_complete');warmup_seconds=time.monotonic()-start
        mx.random.seed(a.seed)
        tok=spm.SentencePieceProcessor(model_file='models/moshi/tokenizer_spm_32k_3.model')
        text_sampler=Sampler(top_k=25,temp=.8) if a.sampling_preset=='offline' else Sampler()
        audio_sampler=Sampler(top_k=250,temp=.8) if a.sampling_preset=='offline' else Sampler()
        gen=LmGen(model,max_steps=codes.shape[1]+5,text_sampler=text_sampler,audio_sampler=audio_sampler)
        text=[];audio=[];frames=[];start=time.monotonic()
        for i in range(codes.shape[1]):
            runtime.check();tick=time.monotonic();result=gen.step(mx.array(codes[:,i][None,:]))
            token=int((result[0] if isinstance(result,tuple) else result).item());agent=gen.last_audio_tokens()
            if agent is not None:audio.append(np.array(agent).astype(np.uint32)[0])
            piece=tok.id_to_piece(token).replace('▁',' ') if token not in (0,3) else '';text.append(piece)
            frames.append(dict(input_frame=i,text_token=token,piece=piece,seconds=time.monotonic()-tick,agent_frame_available=agent is not None,agent_audio_frame_index=len(audio)-1 if agent is not None else None))
            runtime.response_done()
            if i<3 or (i+1)%25==0:
                atomic_json(a.output/'progress.json',dict(completed_frames=i+1,total_frames=codes.shape[1],text=''.join(text),metrics=runtime.check()))
                print(i+1,'/',codes.shape[1],''.join(text)[-180:],flush=True)
        elapsed=time.monotonic()-start
        if not audio:raise RuntimeError('No agent audio generated')
        np.savez_compressed(a.output/'agent_codes.npz',agent_codes=np.stack(audio,axis=1))
        (a.output/'transcript.txt').write_text(''.join(text).strip()+'\n')
        report=dict(input=str(a.input),input_sha256=sha(a.input),adapter=a.adapter,adapter_sha256=sha(a.adapter) if a.adapter else None,seed=a.seed,sampling_preset=a.sampling_preset,input_seconds=codes.shape[1]*.08,original_input_frames=original_frames,explicit_frame_limit=a.max_frames,model_load_seconds=load_seconds,warmup_seconds=warmup_seconds,stream_temporal=a.stream_temporal,wired_limit_bytes=None,text_sampler=asdict(text_sampler),audio_sampler=asdict(audio_sampler),generation_seconds=elapsed,generation_realtime_factor=elapsed/(codes.shape[1]*.08),peak_model_memory_bytes=mx.get_peak_memory(),frames=frames,resource_metrics=runtime.response_done(),scope='Native model-only generation; temporal residency recorded in stream_temporal, no wiring, codec loaded in a separate process. Resource checks add overhead. No quality claim from successful generation.',sampling_seed_policy='Reset after adapter loading and warmup, matching the historical seed protocol.')
        atomic_json(a.output/'metrics.json',report);print(json.dumps({k:v for k,v in report.items() if k!='frames'}),flush=True)

if __name__=='__main__':main()
