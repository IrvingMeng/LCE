import torch
from torch import nn
from losses.arcface_loss import ArcFaceLoss


# modified to use arcface only
def build_losses(config):
    # Build classification loss
    criterion_cla = ArcFaceLoss(scale=config.LOSS.CLA_S, margin=config.LOSS.CLA_M)
    return criterion_cla


def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    # loss /= len(xs)
    return loss

