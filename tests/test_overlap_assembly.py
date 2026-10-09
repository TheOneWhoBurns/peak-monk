"""Synthetic signal fixtures test timing and bookkeeping, not speech quality."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sf=pytest.importorskip('soundfile',reason='Audio assembly tests run in .venv-speech')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stitch_training_overlap import main, digest, choose_boundary


@pytest.mark.parametrize('kind',['barge_in','backchannel'])
def test_word_cut_and_same_speaker_continuation(tmp_path,monkeypatch,kind):
    source=tmp_path/'source';source.mkdir()
    turns=[];aligned=[]
    for i,(role,seconds) in enumerate(zip(['user','assistant','user','assistant'],[2.,10.,1.,1.5])):
        wav=source/f'x-{i}.wav'
        # An active constant signal isolates exact overlap geometry from VAD.
        sf.write(wav,np.full(round(seconds*24000),2000,dtype=np.int16),24000,subtype='PCM_16')
        words=([dict(word=str(j),start=j-.5,end=float(j)) for j in range(1,10)]
               if i==1 else [dict(word='fixture',start=.1,end=.8)])
        content=' '.join(w['word'] for w in words)
        turns.append(dict(role=role,wav=wav.name,content=content,gap_ms=400,
                          duration_seconds=seconds,wav_sha256=digest(wav)))
        aligned.append(dict(wav=wav.name,words=words,reference=content,
                            recognized=content,wer=0.,accepted=True))
    manifest=source/'manifest.jsonl'
    manifest.write_text(json.dumps(dict(id='x',split='train',turns=turns))+'\n')
    audit=source/'x.alignment.json'
    audit.write_text(json.dumps(dict(initial_prompt=None,accepted=True,turns=aligned)))
    draft=tmp_path/'draft.json';draft.write_text(json.dumps(dict(cases=[dict(id='x',kind=kind)])))
    approved=tmp_path/'approved.json';approved.write_text(json.dumps(dict(
        render_manifest_sha256=digest(manifest),draft_sha256=digest(draft),
        accepted_ids=['x'],alignment_sha256={'x':digest(audit)})))
    out=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['stitch_training_overlap','--draft',str(draft),
        '--manifest',str(manifest),'--approved',str(approved),'--output',str(out)])
    original=digest(source/'x-1.wav')
    main()
    report=json.loads((out/'overlap-assembly.json').read_text())['cases'][0]
    assert report['simultaneous_rms_active_seconds']>=.08
    assert report['episode_seconds']<=36
    placements=report['placements']
    assert placements[3]['start_seconds']>=placements[1]['end_seconds']+.2799
    emitted=json.loads((out/'x.alignment.json').read_text())['turns'][1]['words']
    assert digest(source/'x-1.wav')==original
    if kind=='barge_in':
        assert report['boundary_seconds']==5.
        assert max(w['end'] for w in emitted)==5.
        pcm,rate=sf.read(out/'x-1.wav',dtype='int16')
        assert len(pcm)==5*rate and pcm[-1]==0
        assert report['simultaneous_rms_active_seconds']==pytest.approx(.48,abs=.02)
    else:
        assert digest(out/'x-1.wav')==original
        assert placements[1]['end_seconds']-placements[2]['end_seconds']>=2.
        assert len(emitted)==9


def test_short_source_does_not_relax_boundary_gate():
    with pytest.raises(ValueError,match='interior'):
        choose_boundary([dict(end=1.)],3.,1.,'barge_in')
