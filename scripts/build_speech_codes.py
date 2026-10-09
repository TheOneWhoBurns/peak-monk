"""Build separate speaker Mimi streams and measured word-aligned text targets.

Text interleaving follows the official moshi-finetune Interleaver defaults:
main speaker only, 12.5 Hz, padding 3, end-of-padding 0, no BOS/EOS insertion,
and new words replace unfinished token queues (keep_and_shift=False).
Reference: kyutai-labs/moshi-finetune at 2acc879fe7c48f885a18f6cc9548bccb2674d87b.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import rustymimi
import sentencepiece as spm
import soundfile as sf
from stitch_duplex import mix


def interleave_words(words, frames, tokenizer):
    aligned=sorted(words,key=lambda w:w['start'])
    output=np.full(frames,3,dtype=np.int32)
    queue=deque(); i=0; last_end=-1
    for t in range(frames):
        while i<len(aligned) and aligned[i]['start']*12.5<t+1:
            queue=deque(tokenizer.encode(aligned[i]['word'].strip(),add_bos=False,add_eos=False))
            last_end=int(aligned[i]['end']*12.5);i+=1
        if queue:
            if t>0 and output[t-1]==3: output[t-1]=0
            output[t]=queue.popleft()
        elif t<=last_end:
            output[t]=3
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='data/renders/manifest.jsonl')
    p.add_argument('--approved',default='data/audio_approvals.json')
    args=p.parse_args()
    manifest=Path(args.manifest)
    approvals=json.loads(Path(args.approved).read_text())
    rows=[json.loads(l) for l in manifest.read_text().splitlines()]
    rows=[r for r in rows if r['id'] in approvals['accepted_ids']]
    if not rows: raise ValueError('No reviewed, approved audio records')
    weights=Path('models/moshi/tokenizer-e351c8d8-checkpoint125.safetensors')
    with weights.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()=='09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50'
    mimi=rustymimi.Tokenizer(str(weights),num_codebooks=8)
    tokenizer=spm.SentencePieceProcessor(model_file='models/moshi/tokenizer_spm_32k_3.model')
    out=Path('data/speech_codes');out.mkdir(parents=True,exist_ok=True)
    audio_out=Path('data/duplex');audio_out.mkdir(parents=True,exist_ok=True)
    for row in rows:
        path=out/f"{row['id']}.npz"
        if path.exists(): continue
        audit=json.loads((manifest.parent/f"{row['id']}.alignment.json").read_text())
        assert audit['accepted'],row['id']
        corrections=[c for c in approvals.get('word_corrections',[]) if c['id']==row['id']]
        for correction in corrections:
            turn=next(t for t in audit['turns'] if t['wav']==correction['wav'])
            word=turn['words'][correction['word_index']]
            assert word['word']==correction['asr_word']
            word['word']=correction['replacement']
        # TTS manifest uses content; stitcher preserves these verbatim in provenance.
        pcm,placements=mix(row['turns'],manifest.parent)
        sf.write(audio_out/f"{row['id']}.wav",pcm,24000,subtype='PCM_16')
        frames=(len(pcm)+1919)//1920
        padded=np.pad(pcm,((0,frames*1920-len(pcm)),(0,0))).astype(np.float32)/32768
        mimi.reset(); chunks=[]; start=time.monotonic()
        for i in range(frames):
            block=np.ascontiguousarray(padded[i*1920:(i+1)*1920].T[:,None,:])
            encoded=mimi.encode_step(block)
            if encoded is not None: chunks.append(encoded)
        codes=np.concatenate(chunks,axis=-1)
        assert codes.shape==(2,8,frames),codes.shape
        aligned=[]
        for placement,turn in zip(placements,audit['turns']):
            assert placement['wav']==turn['wav']
            if placement['role']=='assistant':
                aligned.extend(dict(w,start=w['start']+placement['start_seconds'],end=w['end']+placement['start_seconds']) for w in turn['words'])
        text=interleave_words(aligned,frames,tokenizer)
        # Agent channels precede user channels exactly as the model expects.
        raw=np.concatenate([text[None,:],codes[0].astype(np.int32),codes[1].astype(np.int32)])
        np.savez_compressed(path,codes=raw)
        info=dict(id=row['id'],split=row['split'],sources=row['sources'],frames=frames,
            seconds=frames*.08,encode_seconds=time.monotonic()-start,
            channel_order='text, agent audio codebooks 0-7, user audio codebooks 0-7',
            text_targets='Reviewed Whisper words with actual estimated timestamps',
            applied_spelling_corrections=corrections,
            codec_sha256='09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50',
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
        print(row['id'],frames,round(info['encode_seconds'],1),flush=True)


if __name__=='__main__': main()
