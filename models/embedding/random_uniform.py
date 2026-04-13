import torch

class XavierUniform(torch.nn.Module):
    """
    Generates embeddings randomly from Gaussian distribution.
    """
    def __init__(self, dim: int):
        """
        Initializes RandomGaussian, sets dimensionality of the generated embedding,
        mean and standard deviation of the Gaussian distribution.
        
        :param dim: Dimensionality of the generated embedding
        :type dim: int
        """
        self.dim = dim
        super().__init__()
    
    def forward(self, N: int):
        """
        Generates random Gaussian embediing with the given number of rows.
        
        :param N: Number of rows in the Gaussian embedding.
        """
        m = torch.empty((N, self.dim), dtype=torch.float)
        torch.nn.init.xavier_uniform_(m)
        return m