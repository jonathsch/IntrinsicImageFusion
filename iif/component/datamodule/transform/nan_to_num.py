import torch


class NanToNumTransform(torch.nn.Module):
    def __init__(self,
                 nan=0,
                 posinf=None,
                 neginf=None):
        super().__init__()
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    def forward(self, x):
        posinf = self.posinf if self.posinf is not None else x[torch.logical_and(torch.isfinite(x), ~torch.isnan(x))].max()
        neginf = self.neginf if self.neginf is not None else x[torch.logical_and(torch.isfinite(x), ~torch.isnan(x))].min()

        x = torch.nan_to_num(x,
                             posinf=posinf,
                             neginf=neginf)
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     raise RuntimeError("Nan or inf values found after nan_to_num")
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
