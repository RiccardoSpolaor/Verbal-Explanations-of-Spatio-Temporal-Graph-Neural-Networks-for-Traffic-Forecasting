"""
Module implementing the error metrics for the training and
validation procedures.
"""
import torch
from torch import nn

def _get_mask(y_true: torch.FloatTensor,
              missing_value: float = 0.) -> torch.FloatTensor:
    """
    Returns a binary mask indicating the positions where the target values
    are not missing.
    
    Arguments
    ---------
    y_true : FloatTensor
        Tensor with the ground truth target values.
    missing_value : float
        Value used to indicate missing values, by default 0.
    
    Returns
    -------
    FloatTensor
        Binary tensor with the same shape as `y_true`
        indicating which values are not missing.
    """
    mask = (y_true != missing_value)
    return mask

def _get_masked_predictions(y_pred: torch.FloatTensor,
                            mask: torch.FloatTensor) -> torch.FloatTensor:
    """
    Masks the predicted values with a binary mask.
    
    Arguments
    ---------
    y_pred : FloatTensor
        Tensor with the predicted target values.
    mask : FloatTensor
        Binary tensor with the same shape as `y_pred` indicating
        which values should be masked.
    
    Returns
    -------
    masked_pred : FloatTensor
        Tensor with the same shape as `y_pred`, with the masked
        values set to 0.
    """
    return y_pred * mask

class MAE(nn.Module):
    """
    Compute the Mean Absolute Error (MAE) between predicted
    and true values, not considering the missing values.
    
    Attributes
    ----------
    mae : L1Loss
        The Absolute Error function. It does not apply any
        reduction to the computed values (e.g.: sum or mean).
    
    Methods
    -------
    forward(y_pred: FloatTensor, y_true: FloatTensor) -> FloatTensor:
        Compute the forward pass of the function.
    """
    def __init__(self) -> None:
        """Initialize the Mean Absolute Error instance."""
        super().__init__()
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, y_pred: torch.FloatTensor,
                y_true: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute the Mean Absolute Error (MAE) between predicted
        and true values, not considering the missing values.
        
        Arguments
        ---------
        y_pred : FloatTensor
            Tensor with the predicted target values.
        y_true : FloatTensor
            Tensor with the true target values.
        
        Returns
        -------
        FloatTensor
            Scalar tensor with the Mean Absolute Error error between
            predicted and true values, not considering the missing
            values.
        """
        mask = _get_mask(y_true)
        y_pred = _get_masked_predictions(y_pred, mask)
        res = self.mae(y_pred, y_true)
        return res.sum() / mask.sum()

class RMSE(nn.Module):
    """
    Compute the Root Mean Square Error (RMSE) between predicted
    and true values, not considering the missing values.
    
    Attributes
    ----------
    mse : MSELoss
        The Square Error function. It does not apply any
        reduction to the computed values (e.g.: sum or mean).
    
    Methods
    -------
    forward(y_pred: FloatTensor, y_true: FloatTensor) -> FloatTensor:
        Compute the forward pass of the function.
    """
    def __init__(self) -> None:
        """Initialize the Root Mean Square Error instance."""
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred: torch.FloatTensor,
                y_true: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute the Root Mean Square Error (RMSE) between predicted
        and true values, not considering the missing values.
        
        Arguments
        ---------
        y_pred : FloatTensor
            Tensor with the predicted target values.
        y_true : FloatTensor
            Tensor with the true target values.
        
        Returns
        -------
        FloatTensor
            Scalar tensor with the Root Mean Square Error between
            predicted and true values, not considering the missing
            values.
        """
        mask = _get_mask(y_true)
        y_pred = _get_masked_predictions(y_pred, mask)
        res = self.mse(y_pred,y_true)
        return torch.sqrt(res.sum() / mask.sum())

class MAPE(nn.Module):
    """
    Compute the Mean Absolute Percentage Error (MAPE) between
    predicted and true values, not considering the missing values.
    
    Attributes
    ----------
    mae : L1Loss
        The Absolute Error function. It does not apply any
        reduction to the computed values (e.g.: sum or mean).
    
    Methods
    -------
    forward(y_pred: FloatTensor, y_true: FloatTensor) -> FloatTensor:
        Compute the forward pass of the function.
    """
    def __init__(self) -> None:
        """Initialize the Mean Absolute Percentage Error instance."""
        super().__init__()
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, y_pred, y_true):
        """
        Compute the Mean Absolute Percentage Error (MAPE) 
        between predicted and true values, not considering the
        missing values.
        
        Arguments
        ---------
        y_pred : FloatTensor
            Tensor with the predicted target values.
        y_true : FloatTensor
            Tensor with the true target values.
        
        Returns
        -------
        FloatTensor
            Scalar tensor with the Mean Absolute Percentage Error
            between predicted and true values, not considering the
            missing values.
        """
        mask = _get_mask(y_true)
        y_pred = _get_masked_predictions(y_pred, mask)
        res = self.mae(y_pred, y_true)
        res = res / y_true.abs()
        res[torch.isinf(res)] = 0.
        res[torch.isnan(res)] = 0.
        return res.sum() / mask.sum()
