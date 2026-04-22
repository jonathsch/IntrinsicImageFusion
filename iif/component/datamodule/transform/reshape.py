import torch


class ReshapeTransform(torch.nn.Module):
    def __init__(self,
                 shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
