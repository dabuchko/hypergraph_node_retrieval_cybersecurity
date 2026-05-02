import torch

def trainable_embeddings_wrapper(num_entries: int, dim: int) -> torch.Tensor:
    """
    Generates a trainable embedding matrix of shape (num_entries, dim) initialized using the
    Xavier uniform initialization as a trainable PyTorch parameter.
    
    :param num_entries: Number of entries (number of rows that should be in the trainable embedding matrix)
    :type num_entries: int
    :param dim: Dimensionality of the embedding (number of columns that should be in the trainable embedding matrix)
    :type dim: int
    :return: Trainable embedding matrix of shape (num_entries, dim) initialized using the
    Xavier uniform initialization as a trainable PyTorch parameter.
    :rtype: Tensor
    """
    x = torch.nn.Parameter(torch.empty((num_entries, dim)), requires_grad=True)
    torch.nn.init.xavier_uniform_(x.data)
    return x