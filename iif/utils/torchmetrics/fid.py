import torch
from torch import Tensor
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics.image.fid import sqrtm
from torchmetrics.utilities import rank_zero_info


def _compute_fid_stable(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6, inf_limit=1e+100) -> Tensor:
    r"""Adjusted version of `Fid Score`_

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular
        inf_limit: limit for treating as singular matrix

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all() or torch.any(covmean > inf_limit):
        rank_zero_info(f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates")
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


class StableFrechetInceptionDistance(FrechetInceptionDistance):
    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid_stable(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)
