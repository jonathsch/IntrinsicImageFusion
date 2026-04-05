import torchvision
import torch


class ToTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_tensor = torchvision.transforms.ToTensor()

    def forward(self, x):
        ndim = len(x.shape)
        if ndim < 4:
            return self.to_tensor(x)
        elif ndim == 4:
            return torch.stack([self.to_tensor(img) for img in x])
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}. Expected 3D or 4D tensor.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
