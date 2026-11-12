import torch

from tokenizer import Tokenizer
from embeddings import TokenEmbedding
from positional_embeddings import PositionalEmbedding
from attention import SelfAttention
from output_projection import OutputProjection

vocab = [
    "<BOS>",
    "<EOS>",
    " ",
    "a",
    "c",
    "e",
    "h",
    "m",
    "o",
    "s",
    "t",
]

tokenizer = Tokenizer(vocab)

text = "the cat"

tokens = tokenizer.encode(text)

print("Text:")
print(text)

print("\nToken IDs:")
print(tokens)

print("\nToken-by-token:")

for character, token_id in zip(text, tokens):
    print(f"{character!r:>4} → {token_id}")


# Convert our Python list into a PyTorch tensor.
token_ids = torch.tensor(tokens)

# Create the embedding table.
embedding = TokenEmbedding(
    vocab_size=len(vocab),
    embedding_dim=8
)

print("\nEmbedding table shape:")
print(embedding.embedding.weight.shape)

print("\nEmbedding table:")
print(embedding.embedding.weight)

# Look up the vector for every token.
vectors = embedding(token_ids)
position_embedding = PositionalEmbedding(
    max_sequence_length=32,
    embedding_dim=8
)

position_vectors = position_embedding(
    sequence_length=len(tokens)
)

combined = vectors + position_vectors
attention = SelfAttention(
    embedding_dim=8,
    head_dim=4
)

(
    Q,
    K,
    V,
    scores,
    scaled_scores,
    masked_scores,
    attention_weights,
    attention_output,
    projected_output
) = attention(combined)

print("\nToken IDs tensor:")
print(token_ids)

print("\nEmbedding shape:")
print(vectors.shape)

print("\nEmbeddings:")
print(vectors)

print("\nPosition embeddings:")
print(position_vectors)

print("\nPosition embedding shape:")
print(position_vectors.shape)

print("\nCombined representation:")
print(combined)

print("\nCombined shape:")
print(combined.shape)

print("\nQ shape:")
print(Q.shape)

print("\nK shape:")
print(K.shape)

print("\nV shape:")
print(V.shape)

print("\nQ:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)

print("\nRaw attention scores:")
print(scores)

print("\nAttention score shape:")
print(scores.shape)

print("\nScaled scores:")
print(scaled_scores)

print("\nMasked scores:")
print(masked_scores)

print("\nAttention weights:")
print(attention_weights)

print("\nAttention weights shape:")
print(attention_weights.shape)

print("\nRow sums:")
print(attention_weights.sum(dim=-1))

# --------------------------------------------------
# ATTENTION OUTPUT
# --------------------------------------------------


#Output projection 
output_projection = OutputProjection(  
    input_dim=4,
    output_dim=8
)
projected_output = output_projection(attention_output)

print("\n" + "=" * 70)
print("OUTPUT PROJECTION") 
print("=" * 70)

print("\nAttention output shape:")
print(attention_output.shape)

print("\nProjected output shape:")
print(projected_output.shape)

print("\nProjected output:")
print(projected_output)

print("\nOutput projection weights:")
print(output_projection.linear.weight)

print("\nOutput projection bias:")
print(output_projection.linear.bias)

print("\n" + "=" * 70)

print("ATTENTION OUTPUT")
print("=" * 70)

print("\nOutput projection weight shape:")
print(attention.output_projection.weight.shape)

print("\nOutput projection bias shape:")
print(attention.output_projection.bias.shape)

print("\nShape:")
print(attention_output.shape)

print("\nMatrix:")
print(attention_output)

print("\n" + "=" * 70)
print("ATTENTION OUTPUT — TOKEN BY TOKEN")
print("=" * 70)

for i, token in enumerate(tokens):
    print(f"\n{token}:")
    print(attention_output[i])

print("\n" + "=" * 70)
print("ATTENTION MAP")
print("=" * 70)

tokens = ["␠" if token == " " else token for token in text]

print("\n        " + " ".join(f"{token:^8}" for token in tokens))

for i, token in enumerate(tokens):
    row = attention_weights[i]

    values = " ".join(
        f"{value.item() * 100:6.1f}%"
        for value in row
    )

    print(f"{token:^8} {values}")