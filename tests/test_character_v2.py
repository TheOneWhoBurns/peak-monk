import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from mlx_lm.tuner.datasets import ChatDataset

ROOT=Path(__file__).resolve().parents[1]


def test_roleplay_rubric_preserves_original_prompts():
    frozen=json.loads((ROOT/'evaluation/roleplay-v2-frozen.json').read_text())
    for path,sha in frozen['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    for name in ('dev','final','speech'):
        old=json.loads((ROOT/f'evaluation/{name}.json').read_text())
        new=json.loads((ROOT/f'evaluation/{name}-roleplay-v2.json').read_text())
        key='user_turns' if name=='speech' else 'turns'
        assert [(r['id'],r[key]) for r in old]==[(r['id'],r[key]) for r in new]


def test_clean_template_has_no_inserted_reasoning_or_prefix_drift():
    tok=AutoTokenizer.from_pretrained(ROOT/'models/stronger-clean')
    row={'messages':[{'role':'user','content':'The latch is stuck.'},
                     {'role':'assistant','content':'Which part will not move?'},
                     {'role':'user','content':'The handle.'},
                     {'role':'assistant','content':'Has it moved before?'}]}
    tokens,offset=ChatDataset([row],tok,mask_prompt=True).process(row)
    prompt=tok.apply_chat_template(row['messages'][:-1],add_generation_prompt=True,return_dict=False)
    assert tokens[:offset]==prompt
    assert tok.decode(tokens[offset:])=='Has it moved before?<|im_end|>\n'
