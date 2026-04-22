import torch


class TakeChannelTransform(torch.nn.Module):
    def __init__(self,
                 channel_to_keep,
                 keep_dim=True):
        super().__init__()
        self.channel_to_keep = channel_to_keep
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x[self.channel_to_keep, ...]
        if self.keep_dim:
            x = x.unsqueeze(0)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
