import torch
from torch import nn

class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, y_pred, y_true):
        return self.mae(y_pred, y_true)

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred,y_true))

class S_MAPELoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # TODO: This is MAE
        #return torch.mean((y_pred - y_true).abs() / (y_true.abs() + 1e-8))
        #abs_diff = torch.abs(y_pred - y_true)
        #error = abs_diff / torch.abs(y_true)
        #error[torch.isinf(error)] = 0.
        # TODO: This is sMAPE, which can handle 0 values unlike MAPE
        divisor = y_true.abs() + y_pred.abs() + self.epsilon
        error = 2*(y_true - y_pred).abs() / divisor
        return torch.mean(error)
