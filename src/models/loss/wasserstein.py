import torch
import torch.nn as nn


######################
# Copyright Yin Li - https://github.com/eelregit/map2map
# Used under GPL-3.0 License
######################


class WDistLoss(nn.Module):
    """Wasserstein distance

    target should have values of 0 (False) or 1 (True)
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return wasserstein_distance_loss(input, target)


def wasserstein_distance_loss(input, target):
    sign = 2 * target - 1

    return - (sign * input).mean()


def wgan_grad_penalty(critic, x, y, lam=10):
    """Calculate the gradient penalty for WGAN
    """
    batch_size = x.shape[0]
    alpha = torch.rand(batch_size).cuda()
    alpha = alpha.reshape(batch_size, *(1,) * (x.dim() - 1))

    xy = alpha * x.detach() + (1 - alpha) * y.detach()

    score = critic(xy.requires_grad_(True))
    # average over spatial dimensions if present
    score = score.flatten(start_dim=1).mean(dim=1)
    # sum over batches because graphs are mostly independent (w/o batchnorm)
    score = score.sum()

    grad, = torch.autograd.grad(
        score,
        xy,
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )

    grad = grad.flatten(start_dim=1)
    penalty = (
        lam * ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
        + 0 * score  # hack to trigger DDP allreduce hooks
    )

    return penalty


# modified from Aladdin Persson 
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/utils.py
def gradient_penalty(critic, real, fake):
    BATCH_SIZE, C, R, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1, 1)).repeat(1, C, R, H, W).cuda()
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
