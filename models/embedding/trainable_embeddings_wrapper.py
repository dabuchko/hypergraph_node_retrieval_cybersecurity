import torch

class TrainableEmbeddingsWrapper(torch.nn.Module):
    def __init__(self, num_entries, in_channels, model):
        super().__init__()
        self.x = torch.nn.Parameter(torch.empty((num_entries, in_channels)))
        torch.nn.init.xavier_uniform_(self.x)
        self.model = model
    def forward(self, x, *args, **kwargs):
        return self.model(self.x, *args, **kwargs)