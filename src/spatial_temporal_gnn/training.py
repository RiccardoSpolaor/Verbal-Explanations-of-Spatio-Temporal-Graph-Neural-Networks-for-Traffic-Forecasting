"""
Module containing the training and validation functions along
with the Checkpoint class and a function to plot the training
history.
"""
import os
from time import time
from typing import Dict, Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from .metrics import MAE, RMSE, MAPE
from .model import SpatialTemporalGNN
from ..data.data_processing import Scaler


class Checkpoint():
    """Class to handle the checkpoints of a model
    
    Arguments
    ---------
    lowest_error : float
        The lowest error of the model predictions obtained by the
        model so far.
    path : str
        The path where the checkpoints will be saved/loaded.
    """
    def __init__(self, path: str, initial_error: float = float('inf')) -> None:
        """Initialize the checkpoint instance.

        Parameters
        ----------
        path : str
            The checkpoint path.
        initial_error : float, optional
            The initial error value, by default inf.
        """
        self.lowest_error = initial_error
        self.path = path
        # Create the path directory if it does not exist.
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def save_best(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                  new_error: float, **kwargs) -> None:
        """
        Possibly save the best model weights and optimizer state 
        in the checkpoint file according to the new value of the
        metric.
        
        Parameters defined in `kwargs` are also saved in the
        checkpoints as a numpy array. 
        
        Parameters
        ----------
        model : Module
            The model which weights are saved.
        optimizer : Optimizer
            The optimizer which state is saved
        new_error : float
            The new error value which is compared to the best so
            far. The checkpoints are updated solely if the new
            error is less than the lowest one saved so far.
        kwargs : Any
            Named arguments saved in the checkpoints as numpy
            arrays.
        """
        if new_error < self.lowest_error:
            # Create the checkpoints
            checkpoint = {}
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['best_error'] = new_error
            for k, v in kwargs.items():
                checkpoint[k] = np.array(v)

            # Save the checkpoints.
            torch.save(checkpoint, self.path)

            # Update the lowest error.
            self.lowest_error = new_error

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
    """
    Trains a spatial temporal graph neural network using the specified
    optimizer and data loaders.
    
    Parameters
    ----------
    model : SpatialTemporalGNN
        The spatial temporal graph neural network to be trained.
    optimizer : Optimizer
        The optimizer used to update the model weights during training.
    train_dataloader : DataLoader
        The data loader for the training set.
    val_dataloader : DataLoader
        The data loader for the validation set.
    scaler : Scaler
        The scaler used to scale the input and output data.
    epochs : int
        The number of epochs to train the model.
    device : str
        The device to run the model on.
    checkpoint : Checkpoint, optional
        A checkpoint momitor for saving the model weights at each epoch, 
        by default None.
    lr_scheduler : object, optional
        A learning rate scheduler to adjust the learning rate during
        training, by default None.
    reload_best_weights : bool, optional
        Whether to reload the best weights of the model after training,
        by default True.
        
    Returns
    -------
    { str: ndarray }
        A dictionary containing the training and validation metrics
        for each epoch, including train and validation MAE, RMSE
        and MAPE.
    """
    # Initialize the training criterions.
    mae_criterion = MAE()
    rmse_criterion = RMSE()
    mape_criterion = MAPE()

    # Initialize the histories.
    metrics = ['train_mae', 'train_rmse', 'train_mape', 'val_mae', 'val_rmse',
               'val_mape']
    history = { m: [] for m in metrics }

    # Set model in training mode.
    model.train()

    # Iterate across the epochs.
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Remove unused tensors from gpu memory.
        torch.cuda.empty_cache()

        # Initialize the running errors.
        running_train_mae = 0.
        running_train_rmse = 0.
        running_train_mape = 0.

        start_time = time()

        for batch_idx, (x, y) in enumerate(train_dataloader):
            # Increment the number of batch steps.
            batch_steps = batch_idx + 1

            # Get the data.
            x = x.type(torch.float32).to(device=device)
            y = y.type(torch.float32).to(device=device)

            x = scaler.scale(x)

            # Compute model predictions.
            y_pred = model(x)
            #print(y_pred.shape, y.shape)

            # Compute the loss on the scaled results and ground truth.
            y_pred = scaler.un_scale(y_pred)

            loss = mae_criterion(y_pred, y)

            # Compute errors and update running errors.
            with torch.no_grad():
                rmse = rmse_criterion(y_pred, y)
                mape = mape_criterion(y_pred, y)

            running_train_mae += loss.item()
            running_train_rmse += rmse.item()
            running_train_mape += mape.item()

            # Zero the gradients.
            optimizer.zero_grad()

            # Use MAE as the loss function for backpropagation.
            loss.backward()

            # Update the weights.
            optimizer.step()

            # Get the batch time.
            epoch_time = time() - start_time
            batch_time = epoch_time / batch_steps

            # Print the batch results.
            print(
                f'[{batch_steps}/{len(train_dataloader)}] -',
                f'{epoch_time:.0f}s {batch_time * 1e3:.0f}ms/step -',

                f'train {{ MAE (loss): {running_train_mae / batch_steps:.3g} -',
                f'RMSE: {running_train_rmse / batch_steps:.3g} -',
                f'MAPE: {running_train_mape * 100. / batch_steps:.3g}% }} -',

                f'lr: {optimizer.param_groups[0]["lr"]:.3g} -',
                f'weight decay: {optimizer.param_groups[0]["weight_decay"]}',
                '             ' if batch_steps < len(train_dataloader) else '',
                end='\r')

        # Set the model in evaluation mode.
        model.eval()

        # Get the average training errors and update the history.
        train_mae = running_train_mae / len(train_dataloader)
        train_rmse = running_train_rmse / len(train_dataloader)
        train_mape = running_train_mape / len(train_dataloader)

        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['train_mape'].append(train_mape)

        # Get the validation results and update the history.
        val_results = validate(model, val_dataloader, scaler, device)
        val_mae, val_rmse, val_mape = val_results

        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_mape'].append(val_mape)

        # Save the checkpoints if demanded.
        if checkpoint is not None:
            err_sum = val_mae + val_rmse + val_mape
            checkpoint.save_best(model, optimizer, err_sum)

        # Print the epoch results.
        print(
            f'[{len(train_dataloader)}/{len(train_dataloader)}] -',
            f'{epoch_time:.0f}s -',

            f'train: {{ MAE (loss): {train_mae:.3g} -',
            f'RMSE: {train_rmse:.3g} -',
            f'MAPE: {train_mape * 100.:.3g}% }} -',

            f'val: {{ MAE: {val_mae:.3g} -',
            f'RMSE: {val_rmse:.3g} -',
            f'MAPE: {val_mape * 100.:.3g}% }} -',

            f'lr: {optimizer.param_groups[0]["lr"]:.3g} -',
            f'weight decay: {optimizer.param_groups[0]["weight_decay"]}')

        # Update the learning rate scheduler.
        lr_scheduler.step(train_mae)

        # Set model in training mode.
        model.train()

    # Load the best weights of the model if demanded.
    if checkpoint is not None and reload_best_weights:
        checkpoint.load_best_weights(model)

    # Set the model in evaluation mode.
    model.eval()

    # Remove unused tensors from gpu memory.
    torch.cuda.empty_cache()

    # Turn the history to numpy arrays.
    for k, v in history.items():
        history[k] = np.array(v)

    return history

def validate(
    model: SpatialTemporalGNN, val_dataloader: DataLoader, scaler: Scaler,
    device: str, n_timestamps_to_predict: Optional[int] = None
    ) -> Tuple[float, float, float]:
    """
    Calculate MAE, RMSE and MAPE scores for The Spatial-Temporal GNN
    on the validation set.

    Arguments
    ---------
    model : SpatialTemporalGNN
        The spatial temporal graph neural network to be trained.
    val_dataloader : DataLoader
        The data loader for the validation set.
    scaler : Scaler
        The scaler used to scale the input and output data.
    device : str
        The device to run the model on.
    n_timestamps_to_predict : int, optional
        Number of timestamps to predict. If None, predict all
        the timestamps. By default None.

    Returns
    -------
    float
        The average MAE score of the predictions on the
        validation set.
    float
        The average RMSE score of the predictions on the
        validation set.
    float
        The average MAPE score of the predictions on the
        validation set.
    """
    torch.cuda.empty_cache()

    # Initialize the validation criterions.
    mae_criterion = MAE()
    rmse_criterion = RMSE()
    mape_criterion = MAPE()

    # Inizialize running errors.
    running_val_mae = 0.
    running_val_rmse = 0.
    running_val_mape = 0.

    with torch.no_grad():
        for _, (x, y) in enumerate(val_dataloader):
            # Get the data.
            x = x.type(torch.float32).to(device=device)
            y = y.type(torch.float32).to(device=device)

            # Scale the input data.
            x = scaler.scale(x)

            # Compute the output.
            y_pred = model(x)

            # Reduce the timestamps to predict if demanded.
            if n_timestamps_to_predict is not None:
                y_pred = y_pred[:, : n_timestamps_to_predict]
                y = y[:, : n_timestamps_to_predict]

            # Un-scale the predictions.
            y_pred = scaler.un_scale(y_pred)

            # Get the prediction errors and update the running errors.
            mae = mae_criterion(y_pred, y)
            rmse = rmse_criterion(y_pred, y)
            mape = mape_criterion(y_pred, y)

            running_val_mae += mae.item()
            running_val_rmse += rmse.item()
            running_val_mape += mape.item()

    # Remove unused tensors from gpu memory.
    torch.cuda.empty_cache()

    # Get the average MAE, RMSE and MAPE scores.
    val_mae = running_val_mae / len(val_dataloader)
    val_rmse = running_val_rmse / len(val_dataloader)
    val_mape = running_val_mape / len(val_dataloader)

    return val_mae, val_rmse, val_mape

def _plot_subplot(
    index: int, train_history: np.ndarray, val_history: np.ndarray,
    title: str, metric_name: str) -> None:
    """Plot a training history subplot for a specific metric.

    Parameters
    ----------
    index : int
        The subplot index.
    train_history : ndarray
        The training history of a specific metric.
    val_history : ndarray
        The validation history of a specific metric.
    title : str
        The title of the subplot.
    metric_name : str
        The name of the considered metric.
    """
    # Plot the subplot at the given index.
    plt.subplot(*index)
    # Set the title.
    plt.title(title)

    # Plot the training and validation history.
    plt.plot(train_history, label='train')
    plt.plot(val_history, label='validation')

    # Set the x and y labels.
    plt.xlabel('epochs')
    plt.ylabel(metric_name)

    # Plot the legend.
    plt.legend()

def plot_training_history(history: Dict[str, np.ndarray]) -> None:
    """Plot the training history of the model.

    Parameters
    ----------
    history : { str: ndarray }
        A dictionary containing the training history values, including:
        * Train MAE (Mean Absolute Error) history.
        * Validation MAE history.
        * Train RMSE (Root Mean Squared Error) history.
        * Validation RMSE history.
        * Train MAPE (Mean Absolute Percentage Error) history.
        * Validation MAPE history.
    """
    plt.figure(figsize=(15, 10))

    # Plot the training history subplots.
    _plot_subplot((2, 1, 1), history['train_mae'], history['val_mae'],
                  'Train and validation MAE history', 'MAE')
    _plot_subplot((2, 2, 3), history['train_rmse'], history['val_rmse'],
                  'Train and validation RMSE history', 'RMSE')
    _plot_subplot((2, 2, 4), history['train_mape'] * 100.,
                  history['val_mape'] * 100.,
                  'Train and validation MAPE history', 'MAPE (%)')

    # Set the title.
    plt.suptitle('Training and validation history', size=16)

    # Configure the layout and plot.
    plt.tight_layout()
    plt.show()
