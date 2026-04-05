import torch
from kornia.filters import bilateral_blur


class BilateralBlurTransform(torch.nn.Module):
    def __init__(self,
                 kernel_size=3,
                 sigma_color=0.1,
                 sigma_space=(1.5, 1.5)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = tuple(sigma_space)

    def forward(self, x):
        x = bilateral_blur(x, self.kernel_size, self.sigma_color, self.sigma_space)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
