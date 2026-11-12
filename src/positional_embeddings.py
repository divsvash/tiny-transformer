import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_sequence_length, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim
        )

    def forward(self, sequence_length):
        positions = torch.arange(sequence_length)

        return self.embedding(positions)