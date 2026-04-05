import torch


class IndexTransform(torch.nn.Module):
    def __init__(self,
                 indices):
        super().__init__()
        self.indices = indices

    def forward(self, x):
        output = []
        for idx in self.indices:
            output += self._get_at_index(x, idx)
        output = torch.stack(output, dim=0)
        return output

    def _get_at_index(self, x, idx):
        idx = tuple(idx)
        if any(i is None for i in idx):
            for dim, idx_val in enumerate(idx):
                if idx_val is not None:
                    x = torch.index_select(x, dim, torch.tensor(idx_val, device=x.device))
            return x
        else:
            return x[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
