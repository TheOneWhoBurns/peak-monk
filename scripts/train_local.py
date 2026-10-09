"""MLX LoRA driver with a corrected assistant-through-EOS objective.

Uses the upstream optimizer, checkpointing and adapter implementation. Explicit
runtime injection leaves the installed library untouched and is tested separately.
"""
import functools
import importlib
from assistant_objective import ThroughEOS, assistant_loss

lora = importlib.import_module('mlx_lm.lora')
original_load = lora.load_dataset


def load_dataset(args, tokenizer):
    datasets = original_load(args, tokenizer)
    return tuple(ThroughEOS(d, tokenizer.eos_token_id) if len(d) else d for d in datasets)


lora.load_dataset = load_dataset
lora.train = functools.partial(lora.train, loss=assistant_loss)

if __name__ == '__main__':
    lora.main()
