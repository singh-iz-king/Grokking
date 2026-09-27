from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class NandaOneLayerTransformer(nn.Module):
    """Faithful one-layer, causal ReLU Transformer used for modular addition."""

    def __init__(
        self,
        modulus: int = 113,
        d_model: int = 128,
        n_heads: int = 4,
        d_head: int = 32,
        d_mlp: int = 512,
        context_length: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if d_model != n_heads * d_head:
            raise ValueError("d_model must equal n_heads * d_head")
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")
        self.modulus = modulus
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_mlp = d_mlp
        self.context_length = context_length
        self.activation = activation
        self.vocab_size = modulus + 1

        self.W_E = nn.Parameter(torch.randn(d_model, self.vocab_size) / math.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(context_length, d_model) / math.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_head, d_model) / math.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, n_heads * d_head) / math.sqrt(d_model))
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.W_U = nn.Parameter(torch.randn(d_model, self.vocab_size) / math.sqrt(self.vocab_size))

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, tokens: Tensor, return_activations: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if tokens.ndim != 2 or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have shape [batch, sequence] within context_length")
        residual = self.W_E[:, tokens].permute(1, 2, 0)
        residual = residual + self.W_pos[: tokens.shape[1]].unsqueeze(0)

        q = torch.einsum("bpd,hmd->bphm", residual, self.W_Q)
        k = torch.einsum("bpd,hmd->bphm", residual, self.W_K)
        v = torch.einsum("bpd,hmd->bphm", residual, self.W_V)
        scores = torch.einsum("bqhm,bkhm->bhqk", q, k) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.causal_mask[: tokens.shape[1], : tokens.shape[1]], -1e10)
        pattern = torch.softmax(scores, dim=-1)
        z = torch.einsum("bhqk,bkhm->bqhm", pattern, v)
        attention_output = torch.einsum("bph,dh->bpd", z.flatten(start_dim=2), self.W_O)
        resid_mid = residual + attention_output

        pre = torch.einsum("bpd,md->bpm", resid_mid, self.W_in) + self.b_in
        post = F.relu(pre) if self.activation == "relu" else F.gelu(pre)
        resid_post = resid_mid + torch.einsum("bpm,dm->bpd", post, self.W_out) + self.b_out
        logits = resid_post @ self.W_U
        if not return_activations:
            return logits
        return logits, {
            "mlp_pre": pre,
            "mlp_post": post,
            "attention_pattern": pattern,
            "resid_mid": resid_mid,
            "resid_post": resid_post,
        }


def neuron_to_logit_map(model: NandaOneLayerTransformer) -> Tensor:
    """Return W_L = W_U W_out, excluding the '=' output class."""
    return model.W_U[:, : model.modulus].T @ model.W_out

