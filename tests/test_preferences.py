import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_preferences import objective,log_probability

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.logits=mx.zeros((4,))
    def __call__(self,x):return mx.broadcast_to(self.logits,(*x.shape,4))


def test_dpo_gradient_prefers_chosen_and_ignores_prompt():
    model=Tiny();chosen=mx.array([[0,3,1]]);rejected=mx.array([[0,3,2]])
    value,grad=nn.value_and_grad(model,objective)(model,chosen,rejected,2,0.,.1,0.)
    mx.eval(value,grad)
    assert abs(value.item()-0.693147)<1e-5
    assert grad['logits'][1].item()<0
    assert grad['logits'][2].item()>0
    assert abs(grad['logits'][0].item())<1e-6
    assert abs(grad['logits'][3].item())<1e-6
    lp,count=log_probability(model,chosen,2)
    assert count.item()==1
    assert abs(lp.item()+1.386294)<1e-5
