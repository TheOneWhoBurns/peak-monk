import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tokenizer_preflight import check_tokenizer


def test_missing_backend_tokenizer_cannot_silently_encode_everything_as_one_token(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'qwen3_5','text_config':{'vocab_size':248320,'eos_token_id':248044}}))
    (tmp_path/'tokenizer_config.json').write_text(json.dumps({'tokenizer_class':'TokenizersBackend','eos_token':'<|im_end|>'}))
    with pytest.raises(ValueError,match='Incomplete tokenizer'):
        check_tokenizer(tmp_path)
