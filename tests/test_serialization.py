"""Catch silent prompt-loss contamination and train/inference template drift."""
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def test_assistant_mask_and_inference_prefix():
    if not (ROOT/'models/base/model.safetensors').exists():
        pytest.skip('Model download not complete')
    from transformers import AutoTokenizer
    from mlx_lm.tuner.datasets import ChatDataset
    tok=AutoTokenizer.from_pretrained(ROOT/'models/base')
    rows=[json.loads(l) for l in (ROOT/'data/prepared/train.jsonl').read_text().splitlines()]
    ds=ChatDataset(rows,tok,mask_prompt=True)
    for r in rows:
        tokens,offset=ds.process(r)
        prompt=tok.apply_chat_template(r['messages'][:-1],add_generation_prompt=True,return_dict=False)
        assert tokens[:offset]==prompt
        assert 0<offset<len(tokens)<=768
        completion=tok.decode(tokens[offset:])
        assert completion==r['messages'][-1]['content']+'<|im_end|>\n'
