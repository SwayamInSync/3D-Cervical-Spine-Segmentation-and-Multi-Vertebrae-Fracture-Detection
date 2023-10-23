import torch
import torch.nn as nn

bce = nn.BCEWithLogitsLoss(reduction='none')


def criterion(logits, targets, device, activated=False):
    if activated:
        losses = nn.BCELoss(reduction='none')(
            logits.view(-1), targets.view(-1))
    else:
        losses = bce(logits.view(-1), targets.view(-1))
    losses[targets.view(-1) > 0] *= 2.
    norm = torch.ones(logits.view(-1).shape[0]).to(device)
    norm[targets.view(-1) > 0] *= 2
    return losses.sum() / norm.sum()
