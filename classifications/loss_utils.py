import torch
import torch.nn as nn


class CECriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.CrossEntropyLoss()

    def forward(self, pred, target_dict):
        target = target_dict['targets']
        return self.crit(pred, target)

class CutmixCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.CrossEntropyLoss()

    def forward(self, pred, target_dict):
        y_a = target_dict['targets_a']
        y_b = target_dict['targets_b']
        lam = target_dict['lam']
        return lam * self.crit(pred, y_a) + (1 - lam) * self.crit(pred, y_b)
