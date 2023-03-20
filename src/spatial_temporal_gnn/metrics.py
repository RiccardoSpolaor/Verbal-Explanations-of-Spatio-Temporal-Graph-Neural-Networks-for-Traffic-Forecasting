import torch
from torch import nn

def _get_mask(y_true: torch.FloatTensor, missing_value: float = 0.) -> torch.FloatTensor:
    mask = (y_true != missing_value)
    return mask

def _get_masked_predictions(y_pred: torch.FloatTensor, mask: torch.FloatTensor) -> torch.FloatTensor:
    return y_pred * mask

class MAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, y_pred, y_true):
        mask = _get_mask(y_true)
        y_pred = _get_masked_predictions(y_pred, mask)
        res = self.mae(y_pred, y_true)
        return res.sum() / mask.sum()

class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true):
        mask = _get_mask(y_true)
        y_pred = _get_masked_predictions(y_pred, mask)
        res = self.mse(y_pred,y_true)
        return torch.sqrt(res.sum() / mask.sum())

class MAPE(nn.Module):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.mae = nn.L1Loss(reduction='none')
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        mask = _get_mask(y_true)
        y_pred = _get_masked_predictions(y_pred, mask)
        res = self.mae(y_pred, y_true)
        res = res / y_true.abs()
        res[torch.isinf(res)] = 0.
        res[torch.isnan(res)] = 0.
        return res.sum() / mask.sum()
