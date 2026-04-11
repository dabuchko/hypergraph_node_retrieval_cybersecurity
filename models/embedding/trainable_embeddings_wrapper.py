import torch

def trainable_embeddings_wrapper(num_entries: int, dim: int) -> torch.Tensor:
    x = torch.nn.Parameter(torch.empty((num_entries, dim)), requires_grad=True)
    torch.nn.init.xavier_uniform_(x.data)
    return x