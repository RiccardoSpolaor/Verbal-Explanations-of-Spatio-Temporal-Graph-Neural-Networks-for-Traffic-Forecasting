import os
from time import time
from typing import Dict, Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from .metrics import MAELoss, RMSELoss, S_MAPELoss
from .model import SpatialTemporalGNN
from ..data.data_processer import Scaler

class Checkpoint():
    """Class to handle the checkpoints of a model."""
    def __init__(self, path: str, initial_error: float = float('inf')) -> None:
        """Initialize the checkpoint instance.
        
        Parameters
        ----------
        path : str
            The checkpoint path.
        initial_error : float, optional
            The initial error value, by default inf.
        """
        self.last_error = initial_error

        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def save_best(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                  new_error: float) -> None:
        """Possibly save the best model weights and optimizer state 
        in the checkpoint file according to the new value of the metric.
        
        Parameters defined in `kwargs` are also saved in the checkpoints
        as an ndarray. 
        
        Parameters
        ----------
        model : Module
            The model which weights are saved.
        optimizer : Optimizer
            The optimizer which state is saved
        new_error : float
            The new error value which is compared to the best so far.
            The checkpoints are updated solely if the new error is less.
        """
        if new_error < self.last_error:
            checkpoint = {}
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['best_error'] = new_error

            torch.save(checkpoint, self.path)

        self.last_error = new_error

    def load_best_weights(self, model: nn.Module) -> None:
        """Load the best weights on a model.

        Parameters
        ----------
        model : Module
            The model for which the best weights are loaded.
        """
        checkpoint = torch.load(self.path)
        model.load_state_dict(checkpoint['model_state_dict'])

def train(
    model: SpatialTemporalGNN, optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader, val_dataloader: DataLoader, scaler: Scaler,
    epochs: int, device: str, checkpoint: Optional[Checkpoint] = None,
    lr_scheduler: Optional[object] = None,
    reload_best_weights: bool = True) -> Dict[str, np.ndarray]:
    # Initialize loss functions.
    mae_function = MAELoss()
    rmse_function = RMSELoss()
    mape_function = S_MAPELoss()

    # Initialize histories.
    metrics = ['train_loss', 'train_mae', 'train_rmse', 'train_smape',
           'val_loss', 'val_mae', 'val_rmse', 'val_smape']
    history = { m: [] for m in metrics }

    # Set model in training mode.
    model.train()

    # Iterate across the epochs.
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Remove unused tensors from gpu memory.
        torch.cuda.empty_cache()

        # Initialize running loss and errors.
        running_train_loss = 0.
        running_train_mae = 0.
        running_train_rmse = 0.
        running_train_mape = 0.

        start_time = time()

        for batch_idx, data in enumerate(train_dataloader, 0):
            # Increment the number of batch steps.
            batch_steps = batch_idx + 1

            # Get the data.
            x, y = data
            x = x.type(torch.float32).to(device=device)
            y = y.type(torch.float32).to(device=device)

            x = scaler.scale(x)

            # Compute model predictions.
            y_pred = model(x)

            # Compute the loss on the scaled results and ground truth.
            loss = mae_function(y_pred, scaler.scale(y))
            running_train_loss += loss.item()

            y_pred = scaler.un_scale(y_pred)
            # Zero the gradients.
            optimizer.zero_grad()

            # Compute errors and update running errors.
            mae = mae_function(y_pred, y)
            rmse = rmse_function(y_pred, y)
            mape = mape_function(y_pred, y)

            running_train_mae += mae.item()
            running_train_rmse += rmse.item()
            running_train_mape += mape.item()

            # Use MAE as the loss function for backpropagation.
            loss.backward()

            # Update the weights.
            optimizer.step()

            epoch_time = time() - start_time
            batch_time = epoch_time / batch_steps

            print(
                f'[{batch_steps}/{len(train_dataloader)}] -',
                f'{epoch_time:.0f}s {batch_time * 1e3:.0f}ms/step -',

                f'train {{ loss: {running_train_loss / batch_steps:.3g} -',
                f'MAE: {running_train_mae / batch_steps:.3g} -',
                f'RMSE: {running_train_rmse / batch_steps:.3g} -',
                f'sMAPE: {running_train_mape * 100. / batch_steps:.3g}% }} -',

                f'lr: {optimizer.param_groups[0]["lr"]:.3g} -',
                f'weight decay: {optimizer.param_groups[0]["weight_decay"]}',
                '             ' if batch_steps < len(train_dataloader) else '',
                end='\r')

        # Set the model in evaluation mode.
        model.eval()

        train_loss = running_train_loss / len(train_dataloader)
        train_mae = running_train_mae / len(train_dataloader)
        train_rmse = running_train_rmse / len(train_dataloader)
        train_smape = running_train_mape / len(train_dataloader)

        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['train_smape'].append(train_smape)

        val_results = validate(model, val_dataloader, mae_function, 
                               rmse_function, mape_function, scaler, device)
        val_loss, val_mae, val_rmse, val_smape = val_results

        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_smape'].append(val_smape)

        if checkpoint is not None:
            err_sum = val_mae + val_rmse + val_smape
            checkpoint.save_best(model, optimizer, err_sum)

        print(
            f'[{len(train_dataloader)}/{len(train_dataloader)}] -',
            f'{epoch_time:.0f}s -',

            f'train: {{ loss: {train_loss:.3g} -',
            f'MAE: {train_mae:.3g} -',
            f'RMSE: {train_rmse:.3g} -',
            f'sMAPE: {train_smape * 100.:.3g}% }} -',

            f'val: {{ loss: {val_loss:.3g} -',
            f'MAE: {val_mae:.3g} -',
            f'RMSE: {val_rmse:.3g} -',
            f'sMAPE: {val_smape * 100.:.3g}% }} -',

            f'lr: {optimizer.param_groups[0]["lr"]:.3g} -',
            f'weight decay: {optimizer.param_groups[0]["weight_decay"]}')

        lr_scheduler.step(train_loss)

        # Set model in training mode.
        model.train()

    if checkpoint is not None and reload_best_weights:
        checkpoint.load_best_weights(model)

    model.eval()

    # Remove unused tensors from gpu memory.
    torch.cuda.empty_cache()

    # Turn histories in ndarrays.
    for k, v in history.items():
        history[k] = np.array(v)

    return history

def validate(
    model: nn.Module, val_dataloader: DataLoader, mae_function: nn.Module,
    rmse_function: nn.Module, mape_function: nn.Module, scaler: Scaler, device: str, 
    n_timestamps_to_predict: Optional[int] = None
    ) -> Tuple[float, float, float, float]:
    torch.cuda.empty_cache()

    running_val_loss = 0.
    running_val_mae = 0.
    running_val_rmse = 0.
    running_val_smape = 0.

    with torch.no_grad():
        for _, data in enumerate(val_dataloader, 0):
            # Get the data.
            x, y = data
            x = scaler.scale(x)
            x = x.type(torch.float32).to(device=device)
            y = y.type(torch.float32).to(device=device)

            # Compute output.
            y_pred = model(x)

            if n_timestamps_to_predict is not None:
                y_pred = y_pred[:, : n_timestamps_to_predict]
                y = y[:, : n_timestamps_to_predict]

            loss = mae_function(y_pred, scaler.scale(y))
            running_val_loss += loss.item()

            y_pred = scaler.un_scale(y_pred)

            # Loss
            mae = mae_function(y_pred, y)
            rmse = rmse_function(y_pred, y)
            mape = mape_function(y_pred, y)
            running_val_mae += mae.item()
            running_val_rmse += rmse.item()
            running_val_smape += mape.item()

    torch.cuda.empty_cache()

    val_loss = running_val_loss / len(val_dataloader)
    val_mae = running_val_mae / len(val_dataloader)
    val_rmse = running_val_rmse / len(val_dataloader)
    val_smape = running_val_smape / len(val_dataloader)

    return val_loss, val_mae, val_rmse, val_smape
