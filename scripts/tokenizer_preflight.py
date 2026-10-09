"""Fail before loading model weights when a local tokenizer is incomplete."""
import json
from pathlib import Path
from transformers import AutoTokenizer


def check_tokenizer(model_path):
    root=Path(model_path)
    cfg=json.loads((root/'config.json').read_text())
    cfg=cfg.get('text_config',cfg)
    tok=AutoTokenizer.from_pretrained(root,local_files_only=True)
    text='The kettle is warm, and someone is walking up the path.'
    ids=tok.encode(text,add_special_tokens=False)
    if tok.vocab_size<1000 or len(ids)<3 or not any(i not in tok.all_special_ids for i in ids):
        raise ValueError('Incomplete tokenizer: ordinary text does not produce a nontrivial token sequence')
    if tok.decode(ids)!=text:
        raise ValueError('Tokenizer failed a plain-text round trip')
    expected=cfg.get('eos_token_id')
    generation=root/'generation_config.json'
    if generation.exists():
        expected=json.loads(generation.read_text()).get('eos_token_id',expected)
    if expected is not None and tok.eos_token_id not in (expected if isinstance(expected,list) else [expected]):
        raise ValueError('Tokenizer EOS does not match model configuration')
    if max(ids)>=cfg['vocab_size']:
        raise ValueError('Tokenizer IDs exceed model vocabulary')
    return dict(vocab_size=tok.vocab_size,eos_token_id=tok.eos_token_id,round_trip=True)
