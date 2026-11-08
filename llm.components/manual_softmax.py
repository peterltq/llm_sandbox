import torch

def softmax(x, dim=-1):
    max_values = torch.max(x, dim=dim, keepdim=True).values
    shifted_x = x - max_values

    exponentials = torch.exp(shifted_x)
    denominator = torch.sum(exponentials, dim=dim, keepdim=True)

    return exponentials / denominator
