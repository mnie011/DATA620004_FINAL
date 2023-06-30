import torch
import torch.nn as nn
import torch.nn.functional as F


class CECriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.CrossEntropyLoss()

    def forward(self, pred, target_dict):
        target = target_dict['targets']
        return self.crit(pred, target)


class VAECriterion(nn.Module):
    def __init__(self, kld_weight):
        super().__init__()
        self.kld_weight = kld_weight

    def forward(self, recons, targets, mu, log_var):
        recons_loss = F.mse_loss(recons, targets)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.kld_weight * kld_loss
        return loss


class CutmixCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.CrossEntropyLoss()

    def forward(self, pred, target_dict):
        y_a = target_dict['targets_a']
        y_b = target_dict['targets_b']
        lam = target_dict['lam']
        return lam * self.crit(pred, y_a) + (1 - lam) * self.crit(pred, y_b)
