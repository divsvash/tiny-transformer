import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            head_dim,
            bias=False
        )

        # Output projection
        self.output_projection = nn.Linear(
            head_dim,
            embedding_dim
        )

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.T

        scale = K.size(-1) ** 0.5

        scaled_scores = scores / scale

        mask = torch.tril(
            torch.ones(
                scaled_scores.shape,
                dtype=torch.bool
            )
        )

        masked_scores = scaled_scores.masked_fill(
            ~mask,
            float("-inf")
        )

        attention_weights = torch.softmax(
            masked_scores,
            dim=-1
        )

        # Weighted combination of V
        attention_output = attention_weights @ V

        # Project back from head_dim → embedding_dim
        projected_output = self.output_projection(
            attention_output
        )

        return (
            Q,
            K,
            V,
            scores,
            scaled_scores,
            masked_scores,
            attention_weights,
            attention_output,
            projected_output
        )