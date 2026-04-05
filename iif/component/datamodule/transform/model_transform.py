import torch


class ModelTransform(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
