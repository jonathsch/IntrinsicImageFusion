import torch


class Clamp(torch.nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x) -> torch.Tensor:
        """
        Transforms the range of tensor.
        :param x: The input tensor
        :return: The transformed tensor
        """
        return torch.clamp(x, min=self.min, max=self.max)

    def inverse(self, y):
        """
        Inverse transforms the range of tensor.
        :param y: The transformed tensor
        :return: The inverse transformed tensor
        """
        raise NotImplementedError(f"Clamping is not inversible")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min}, max={self.max}, linear={self.linear})"
