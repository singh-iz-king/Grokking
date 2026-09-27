import torch

from src.model import NandaOneLayerTransformer


def test_transformer_shapes() -> None:
    model = NandaOneLayerTransformer(
        modulus=11, d_model=16, n_heads=4, d_head=4, d_mlp=32
    )
    logits, activations = model(torch.tensor([[1, 2, 11], [4, 9, 11]]),
                                return_activations=True)
    assert logits.shape == (2, 3, 12)
    assert activations["mlp_pre"].shape == (2, 3, 32)
    assert activations["mlp_post"].shape == (2, 3, 32)
    assert activations["attention_pattern"].shape == (2, 4, 3, 3)

