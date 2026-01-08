import torch

class MatrixFactorization(torch.nn.Module):
    """
    MatrixFactorization utilizes Non-negative Matrix Factorization approach
    to generate embeddings for the rows of the passed matrix.
    """
    def __init__(self, dim: int=128, weight_decay: float=0.1, iterations: int=None, delta: float=1e-4):
        """
        Initializes MatrixFactorization class.
        
        :param dim: Target dimensionality of the generated embedding matrix.
        :type dim: int
        :param weight_decay: Weight decay to be applied during each factorization iteration.
        :type weight_decay: float
        :param iterations: Maximum number of iterations. Set None if unlimited.
        :type iterations: int
        :param delta: The lower threshold for the distance of factor matrices
        compared to the previous iteration. May be unset (set to None) only if
        the maximum number of iterations is set.
        :type delta: float
        """
        # verify input
        if not isinstance(dim, int) or dim<=0:
            raise "The number of dimensions must be a positive integer."
        if iterations!=None and (not isinstance(iterations, int) or iterations<=0):
            raise "The number of iterations must be a positive integer."
        if not isinstance(delta, float) and not isinstance(delta, int) or delta<=0:
            if iterations==None:
                raise "The number of iterations is not set and delta is in invalid format. "
                "Set delta to be positive float, otherwise algorithm will not terminate."
            else:
                delta = 0
        if not isinstance(weight_decay, float) and not isinstance(weight_decay, int):
            raise "Weight decay rate must be a float."
        self.dim = dim
        self.iterations = iterations
        self.delta = delta
        self.weight_decay = weight_decay
        super().__init__()
    
    def forward(self, x: torch.Tensor, column_weights: torch.Tensor = None):
        """
        Generates row embedding for passed sparse binary matrix x.
        
        :param x: Sparse binary matrix. The matrix is converted to boolean automatically.
        All values that differ from zero will be treated equally.
        :type x: torch.Tensor
        :param column_weights: The weight of each column. If no weights are provided,
        the weight 1 is assumed for all columns.
        :type column_weights: torch.Tensor
        """
        if column_weights==None:
            column_weights = torch.ones((x.shape[1],), dtype=torch.float)
        column_weights = column_weights.reshape(1, -1)
        # initialize factors
        P = torch.empty((x.shape[0], self.dim), dtype=torch.float)
        Q = torch.empty((x.shape[1], self.dim), dtype=torch.float)
        torch.nn.init.xavier_uniform_(P)
        torch.nn.init.xavier_uniform_(Q)
        P_old = P.clone()
        Q_old = Q.clone()
        # this is required to perform first iteration regardless of delta value
        P_old.fill_(torch.inf)
        Q_old.fill_(torch.inf)
        delta = self.delta
        iteration = 0
        x = x.bool().float() # leave only 0/1 values
        column_weights = column_weights.float()

        # compute P and Q through multiple iterations until maximum number
        # of iterations is performed or any of the factors changed by value smaller than delta
        while (self.iterations==None or iteration<self.iterations) and (min((P-P_old).abs().mean().item(), (Q-Q_old).abs().mean().item()) > delta):
            P_old = P.clone()
            Q_old = Q.clone()
            iteration += 1
            # compute P
            P = torch.inverse((Q.T * column_weights) @ Q) @ Q.T
            P = P * column_weights @ x.T
            P = P.T
            P -= self.weight_decay * P
            # compute Q
            Q = (torch.inverse(P.T @ P) @ P.T @ x).T
            Q -= self.weight_decay * Q
        
        return P, Q