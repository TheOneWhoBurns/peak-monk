"""Use Qwen3's native non-thinking format consistently in SFT and generation."""
import json
from pathlib import Path
import shutil
from transformers import AutoTokenizer

root=Path(__file__).resolve().parents[1]
source=root/'models/reference'
dest=root/'models/instruct'
dest.mkdir(parents=True,exist_ok=True)
for path in source.iterdir():
    if path.is_file() and path.suffix in ('.json','.txt','.jinja'):
        shutil.copyfile(path,dest/path.name)
weights=dest/'model.safetensors'
if not weights.exists(): weights.symlink_to('../reference/model.safetensors')
tok=AutoTokenizer.from_pretrained(source)
template='{% set enable_thinking = false %}\n'+tok.chat_template
(dest/'chat_template.jinja').write_text(template)
config=json.loads((dest/'tokenizer_config.json').read_text())
config['chat_template']=template
(dest/'tokenizer_config.json').write_text(json.dumps(config,indent=2)+'\n')
print('Prepared native non-thinking training template:',dest)
