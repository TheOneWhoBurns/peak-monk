import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mlx.core as mx
from assistant_objective import ThroughEOS, assistant_loss


def test_no_loss_on_post_sequence_padding():
    class Model:
        def __call__(self, batch):
            return mx.zeros((*batch.shape, 16))
    # Assistant starts at token 2, EOS is token 3; indices 4 and 5 are padding.
    batch = mx.array([[2, 7, 9, 5, 0, 0]])
    loss, count = assistant_loss(Model(), batch, mx.array([[2, 4]]))
    assert count.item() == 2
    assert abs(loss.item() - 2.772588722) < 1e-6


def test_crop_after_final_assistant_eos_not_history_eos():
    class Data:
        def process(self, item):
            return [2,5,7,9,5,10], 3
    tokens, offset = ThroughEOS(Data(), 5).process({})
    assert tokens == [2,5,7,9,5] and offset == 3
