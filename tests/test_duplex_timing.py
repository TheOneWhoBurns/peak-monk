import importlib.util
from pathlib import Path
import wave
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('stitch',Path(__file__).resolve().parents[1]/'scripts/stitch_duplex.py')
stitch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stitch)

def render(path,seconds,value):
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000)
        f.writeframes(np.full(round(seconds*24000),value,dtype='<i2').tobytes())

def test_backchannel_preserves_long_speaker_and_channel_identity(tmp_path):
    render(tmp_path/'user.wav',2,1000)
    render(tmp_path/'agent.wav',.2,2000)
    output,annotations=stitch.mix([
        {'role':'user','wav':'user.wav','text':'long','gap_ms':0},
        {'role':'assistant','wav':'agent.wav','text':'yes','gap_ms':-1500}],tmp_path)
    assert annotations[1]['start_seconds']==.5
    assert len(output)==3*24000
    assert np.all(output[12000:16800,0]==2000)
    assert np.all(output[:48000,1]==1000)
    assert np.all(output[16800:,0]==0)

def test_invalid_speaker_overlap_is_rejected(tmp_path):
    render(tmp_path/'a.wav',1,1000)
    turns=[{'role':'assistant','wav':'a.wav','text':'x','gap_ms':0},
           {'role':'assistant','wav':'a.wav','text':'y','gap_ms':-100}]
    with pytest.raises(ValueError,match='same speaker'):
        stitch.mix(turns,tmp_path)
