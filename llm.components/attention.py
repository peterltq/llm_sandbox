import math
import torch

def attention(q, k, v, mask=None):
    # q: [batch, heads, query_length, head_dim]
    # k: [batch, heads, key_length, head_dim]
    # v: [batch, heads, key_length, head_dim]

    scores = torch.matmul(q, k.transpose(-2, -1))
    print(f"qk^t: {scores}")
    scores = scores / math.sqrt(q.size(-1))
    print(f"Div sqrt(D): {scores}")

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)

    output = torch.matmul(probabilities, v)
    return output
