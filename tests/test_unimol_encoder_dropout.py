import torch
from loom.oracles.uniqsar_oracle import PureTransformerEncoderWithPair


def test_dropout_only_changes_training_and_preserves_checkpoint_contract():
    torch.manual_seed(42)
    model=PureTransformerEncoderWithPair(2,8,16,2,dropout=.2)
    legacy=PureTransformerEncoderWithPair(2,8,16,2)
    legacy.load_state_dict(model.state_dict(),strict=True)
    x=torch.randn(2,5,8);bias=torch.randn(4,5,5);pad=torch.tensor([[False]*5,[False]*3+[True]*2])
    model.eval();legacy.eval()
    assert torch.equal(model(x,bias,pad),legacy(x,bias,pad))
    model.train();a=model(x,bias,pad);b=model(x,bias,pad)
    assert not torch.equal(a,b)
    a[:,0,0].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
