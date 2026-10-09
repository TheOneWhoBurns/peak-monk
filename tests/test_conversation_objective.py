import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mlx.core as mx
from conversation_objective import encode_conversation, windows, conversation_loss


def test_conversation_masks_match_each_individual_target():
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    tok = AutoTokenizer.from_pretrained(root / 'models/stronger-clean', local_files_only=True)
    messages = [{'role':'user','content':'Hello.'}, {'role':'assistant','content':'Jun here.'}, {'role':'user','content':'What are you doing?'}, {'role':'assistant','content':'Mending a strap.'}]
    tokens, mask = encode_conversation({'messages':messages}, tok)
    actual = [t for t, m in zip(tokens, mask) if m]
    expected = []
    for i in (1, 3):
        prefix = tok.apply_chat_template(messages[:i], add_generation_prompt=True, return_dict=False)
        full = tok.apply_chat_template(messages[:i + 1], return_dict=False)
        target = full[len(prefix):]
        expected.extend(target[:target.index(tok.eos_token_id) + 1])
    assert actual == expected
    assert actual.count(tok.eos_token_id) == 2
    assert not mask[0]


def test_window_boundaries_preserve_every_target_exactly_once():
    tokens = list(range(450))
    mask = [i % 7 in (1, 2, 3) for i in tokens]
    chunks = list(windows(tokens, mask, 160, 32))
    actual = [t for ts, ms in chunks for t, m in zip(ts, ms) if m]
    assert actual == [t for t, m in zip(tokens, mask) if m]
    assert all(len(ts) <= 160 for ts, _ in chunks)


def test_all_turn_loss_ignores_caller_and_padding_gradients():
    logits = mx.zeros((1, 6, 16))
    batch = mx.array([[1, 2, 3, 4, 5, 6, 0]])
    mask = mx.array([[False, False, True, True, False, True, False]])
    def objective(values):
        return conversation_loss(lambda _: values, batch, mask)[0]
    gradient = mx.grad(objective)(logits)
    assert mx.all(gradient[:, [0, 3, 5], :] == 0).item()
    assert mx.any(gradient[:, [1, 2, 4], :] != 0).item()
    assert conversation_loss(lambda _: logits, batch, mask)[1].item() == 3
