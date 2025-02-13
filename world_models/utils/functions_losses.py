import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from einops import rearrange, reduce
from torch.distributions import MultivariateNormal
from world_models.modules.VAE.vae_base import BaseDistributionParams, CategoricalDistributionParams, GaussianDistributionParams


@torch.no_grad()
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.no_grad()
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class SymLogLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = MSELoss()

    def forward(self, output, target):
        # target = symlog(target)
        return 0.5 * self.mse_loss(output, target) # 0.5*F.mse_loss(output, target)


class SymLogTwoHotLoss(nn.Module):
    def __init__(self, num_classes, lower_bound, upper_bound):
        super().__init__()
        self.num_classes = num_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_length = (upper_bound - lower_bound) / (num_classes-1)

        # use register buffer so that bins move with .cuda() automatically
        self.bins: torch.Tensor
        self.register_buffer(
            'bins', torch.linspace(-20, 20, num_classes), persistent=False)

    def forward(self, output, target):
        target = symlog(target)
        assert target.min() >= self.lower_bound and target.max() <= self.upper_bound

        index = torch.bucketize(target, self.bins)
        diff = target - self.bins[index-1]  # -1 to get the lower bound
        weight = diff / self.bin_length
        weight = torch.clamp(weight, 0, 1)
        weight = weight.unsqueeze(-1)

        target_prob = (1-weight)*F.one_hot(index-1, self.num_classes) + weight*F.one_hot(index, self.num_classes)

        loss = -target_prob * F.log_softmax(output, dim=-1)
        loss = loss.sum(dim=-1)
        return loss.mean()

    def decode(self, output):
        return symexp(F.softmax(output, dim=-1) @ self.bins)
    
class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class GaussianKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_mean, p_var, q_mean, q_var):
        """
        Compute the KL divergence between two Gaussian distributions with free bits.

        Args:
            p_mean: Mean vector of the first Gaussian distribution (batch, dim).
            p_var: Variance vector (diagonal of covariance matrix) of the first Gaussian distribution (batch, dim).
            q_mean: Mean vector of the second Gaussian distribution (batch, dim).
            q_var: Variance vector (diagonal of covariance matrix) of the second Gaussian distribution (batch, dim).

        Returns:
            kl_div: KL divergence with free bits applied.
            real_kl_div: Raw KL divergence before applying free bits.
        """
        # Ensure variances are positive
        p_var = torch.clamp(p_var, min=1e-6)
        q_var = torch.clamp(q_var, min=1e-6)

        # Compute KL divergence for each dimension
        log_term = torch.log(q_var) - torch.log(p_var)
        trace_term = p_var / q_var
        mean_diff_term = (p_mean - q_mean) ** 2 / q_var
        kl_per_dim = 0.5 * (trace_term + mean_diff_term - 1 + log_term)

        # Sum over dimensions
        kl_div = kl_per_dim.sum(dim=-1)  # Sum over dimensions
        kl_div = kl_div.mean()  # Average over batch
        real_kl_div = kl_div

        # Apply free bits threshold
        kl_div = torch.max(torch.ones_like(kl_div) * self.free_bits, kl_div)

        return kl_div, real_kl_div

class UniversalKLLoss(nn.Module):
    def __init__(self, free_bits=0.0):
        super().__init__()
        self.free_bits = free_bits
        self.categorical_kl_loss = CategoricalKLDivLossWithFreeBits(free_bits)
        self.gaussian_kl_loss = GaussianKLDivLossWithFreeBits(free_bits)

    def forward(self, p_params: BaseDistributionParams, q_params: BaseDistributionParams):
        if not isinstance(p_params, type(q_params)):
            raise ValueError(f"Distributions must be of the same type for KL divergence calculation. p_params:{type(p_params)},q_params:{type(q_params)} ")

        if isinstance(p_params, CategoricalDistributionParams):
            return self.categorical_kl_loss(p_params.logits(), q_params.logits())
        elif isinstance(p_params, GaussianDistributionParams):
            p_mean, p_logvar = p_params.params
            q_mean, q_logvar = q_params.params
            p_var = torch.exp(p_logvar)
            q_var = torch.exp(q_logvar)
            return self.gaussian_kl_loss(p_mean, p_var, q_mean, q_var)
        else:
            raise ValueError(f"Unsupported distribution type: {type(p_params)}")

if __name__ == "__main__":
    loss_func = SymLogTwoHotLoss(255, -20, 20)
    output = torch.randn(1, 1, 255).requires_grad_()
    target = torch.ones(1).reshape(1, 1).float() * 0.1
    print(target)
    loss = loss_func(output, target)
    print(loss)

    # prob = torch.ones(1, 1, 255)*0.5/255
    # prob[0, 0, 128] = 0.5
    # logits = torch.log(prob)
    # print(loss_func.decode(logits), loss_func.bins[128])
