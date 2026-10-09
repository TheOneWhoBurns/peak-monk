"""Fetch a pinned, data-only MLX conversion; install explicit non-thinking ChatML."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "johnlakness/Qwen3-1.7B-Base-mlx-4Bit"
REVISION = "ccbac499d70a93105c0e96618dd40564f577afa2"
dest = ROOT / "models/base"
snapshot_download(REPO, revision=REVISION, local_dir=dest,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md"],
                  max_workers=2)
template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
(dest / "chat_template.jinja").write_text(template)
config = json.loads((dest / "tokenizer_config.json").read_text())
config["chat_template"] = template
config["eos_token"] = "<|im_end|>"
(dest / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
gen = json.loads((dest / "generation_config.json").read_text())
gen["eos_token_id"] = 151645
(dest / "generation_config.json").write_text(json.dumps(gen, indent=2))
(ROOT / "model_provenance.json").write_text(json.dumps({
    "repo": REPO, "revision": REVISION, "upstream": "Qwen/Qwen3-1.7B-Base",
    "license": "Apache-2.0", "modifications": "Non-thinking ChatML serialization and im_end EOS; weights unchanged",
    "weights": "4-bit MLX community conversion; upstream attribution verified from model card, not bitwise re-converted"
}, indent=2) + "\n")
print("Model ready:", dest, flush=True)
