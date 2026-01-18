import torch

class RandomGaussian(torch.nn.Module):
    """
    Generates embeddings randomly from Gaussian distribution.
    """
    def __init__(self, dim:int, mean:float=0, std:float=1):
        """
        Initializes RandomGaussian, sets dimensionality of the generated embedding,
        mean and standard deviation of the Gaussian distribution.
        
        :param dim: Dimensionality of the generated embedding
        :type dim: int
        :param mean: Mean of the Gaussian distribution
        :type mean: float
        :param std: Standard deviation of the Gaussian distibution
        :type std: float
        """
        self.dim = dim
        self.mean = mean
        self.std = std
        super().__init__()
    
    def forward(self, N: int):
        """
        Generates random Gaussian embediing with the given number of rows.
        
        :param N: Number of rows in the Gaussian embedding.
        """
        m = torch.empty((N, self.dim), dtype=torch.float)
        torch.nn.init.normal_(m, self.mean, self.std)
        return m