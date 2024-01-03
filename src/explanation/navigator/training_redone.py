"""
from math import ceil
from time import time
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.spatial_temporal_gnn.model import SpatialTemporalGNN
from src.spatial_temporal_gnn.metrics import MAE, RMSE, MAPE
from src.spatial_temporal_gnn.training import Checkpoint
from src.explanation.navigator.model import Navigator
from src.data.data_processing import Scaler

def _simulate_model(
    instance: torch.FloatTensor, events_scores: torch.FloatTensor
    ) -> torch.FloatTensor:
    # Apply the sigmoid function to the events scores.
    events_scores = events_scores.sigmoid()
    # Get a random uniform tensor in order to apply the differentiable
    # relaxed Bernoulli function.
    eps = torch.rand_like(events_scores)
    # Compute the relaxed Bernoulli distribution and apply the sigmoid
    # function to the resulting scores.
    events_scores = torch.sigmoid((eps.log() - (1 - eps).log() + events_scores) / 2.0)
    # TODO: Simulate all events, not just the speed events.
    # Threshold the events scores on 0.5.
    result = events_scores >= .5
    # Mask the instance according to the thresholded event scores.
    instance = result * instance
    return instance

def train(
    model: Navigator, optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader, val_dataloader: DataLoader,
    spatial_temporal_gnn: SpatialTemporalGNN, scaler: Scaler,
    epochs: int, validations_per_batch: int = 1,
    checkpoint: Optional[Checkpoint] = None,
    lr_scheduler: Optional[object] = None,
    reload_best_weights: bool = True) -> Dict[str, np.ndarray]:
    # Get the device that is used for training and querying the model.
    device = model.device
    
    # Set the valdation step inside the batch
    assert validations_per_batch > 0, \
        'The number of validations per batch must be greater than zero.'
    val_step = ceil(len(train_dataloader) / validations_per_batch)

    # Initialize the training criterions.
    ae_criterion = MAE(apply_average=False)
    mae_criterion = torch.nn.L1Loss(reduction='mean')
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

        for batch_idx, (x, input_events, target_event, y) in enumerate(
            train_dataloader):
            # Increment the number of batch steps.
            batch_steps = batch_idx + 1

            # Create a batch of simulated instances each having a different
            # speed event removed.
            x_with_removed_speed_events = []

            for e in input_events:
                timestep = int(e[0].item())
                node = int(e[1].item())
                # Remove the speed event at timestep i and node j.
                x_simulated = x.clone()
                x_simulated[timestep, node] = 0.

                x_with_removed_speed_events.append(x_simulated)

            # Create a batch of simulated instances values.
            # Create a dataloader for the simulated instances.
            with torch.no_grad():
                simulated_instances_loader = DataLoader(
                    x_with_removed_speed_events, batch_size=64,
                    shuffle=False)

                simulated_instances_scores = []
                for simulated_batch in simulated_instances_loader:
                    simulated_batch = scaler.scale(simulated_batch)
                    simulated_batch = simulated_batch.float().to(device=device)
                    y_pred = spatial_temporal_gnn(simulated_batch)
                    y_pred = scaler.un_scale(y_pred)

                    # Repeat the target graph y by adding a batch dimension for the y_pred batch size.
                    y_repeated = y.unsqueeze(0).repeat(y_pred.shape[0], 1, 1, 1).float().to(device=device)

                    simulated_instances_scores.append(ae_criterion(y_pred, y_repeated))
                simulated_instances_scores = torch.cat(simulated_instances_scores).unsqueeze(-1)
            # Repeat the target event for the y_pred batch size.

            # Create a score of all the simulated instances inversely proportional to the mae_criterion.
            max_ae = simulated_instances_scores.max()
            min_ae = simulated_instances_scores.min()
            simulated_instances_scores = (max_ae - simulated_instances_scores) / (max_ae - min_ae)
            # Here build the input events and their correlation to the target event and compute regression.
            
            # target_scores = torch.zeros_like((input_events.shape[0], 1))

            # Get the data.
            x = x.float().to(device=device)
            input_events = input_events.float().to(device=device)
            y = y.float().to(device=device)
            target_event = target_event.float().to(device=device)

            # Repeat the target event for all the input events.
            target_repeated = target_event.unsqueeze(0).repeat(
                input_events.shape[0], 1)

            # Compute the correlation scores between each input event and
            # the target event.
            event_scores = model(input_events, x, target_repeated)

            # Compute the MAE loss.
            loss = mae_criterion(event_scores, simulated_instances_scores)

            # Compute the RMSE and MAPE metrics.
            with torch.no_grad():
                rmse = torch.tensor(0.)
                mape = torch.tensor(0.)

            # Update the running errors.
            running_train_mae += loss.item()
            running_train_rmse += rmse.item()
            running_train_mape += mape.item()

            # Zero the gradients.
            optimizer.zero_grad()

            # Use MAE as the loss function for backpropagation.
            loss.backward()

            # Zero the gradients of the Spatial-Temporal GNN model.
            #for param in spatial_temporal_gnn.parameters():
            #    param.grad[:] = 0

            # Update the weights.
            optimizer.step()

            # Get the batch time.
            epoch_time = time() - start_time
            batch_time = epoch_time / batch_steps

            # Check if it is time to apply validation.
            apply_validation = (batch_idx + 1) % val_step == 0
            
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
                end='\r' if not apply_validation else '\n')

            # Evaluate on validation set.
            if apply_validation:
                # Set the model in eval mode.
                model.eval()

                # Remove unused tensors from gpu memory.
                torch.cuda.empty_cache()

                # Compute the validation scores.
                '''val_results = validate(model, val_dataloader, spatial_temporal_gnn,
                                    scaler)
                val_mae, val_rmse, val_mape = val_results

                # Remove unused tensors from gpu memory.
                torch.cuda.empty_cache()
                
                # Print the validation step results.
                print(
                    '\t'
                    f'val step -',

                    f'val: {{ MAE: {val_mae:.3g} -',
                    f'RMSE: {val_rmse:.3g} -',
                    f'MAPE: {val_mape * 100.:.3g}% }} -',

                    f'lr: {optimizer.param_groups[0]["lr"]:.3g} -',
                    f'weight decay: {optimizer.param_groups[0]["weight_decay"]}'
                    )

                # Save the checpoints.
                if checkpoint is not None:
                    err_sum = val_mae + val_rmse + val_mape
                    checkpoint.save_best(model, optimizer, err_sum)'''

                # Set the model in train mode.
                model.train()

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
        val_results = validate(model, val_dataloader, spatial_temporal_gnn,
                               scaler)
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
        #lr_scheduler.step(train_mae)

        # Set model in training mode.
        lr_scheduler.step()
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
    model: Navigator, val_dataloader: DataLoader,
    spatial_temporal_gnn: SpatialTemporalGNN, scaler: Scaler
    ) -> Tuple[float, float, float]:
    device = model.device
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
        for x, input_events, target_event, y in val_dataloader:
            # Get the data.
            x = x.type(torch.float32).to(device=device)
            input_events = input_events.type(torch.float32).to(device=device)
            y = y.type(torch.float32).to(device=device)
            
            # Repeat the target event for all the input events.
            target_repeated = target_event.unsqueeze(0).repeat(
                input_events.shape[0], 1).type(torch.float32).to(device=device)

            # Compute the correlation scores between each input event and
            # the target event.
            event_scores = model(input_events, target_repeated)
            # Get a score mask.
            score_mask = torch.zeros((x.shape[0], x.shape[1], 1))
            # Set the scores in the score mask.
            for i, event in enumerate(input_events):
                timestep = int(event[0].item())
                node = int(event[1].item())
                score_mask[timestep, node, 0] = event_scores[i]
            # Move the score mask to the device.
            score_mask = score_mask.to(device=device)
            # Simulate the input instance according to the score mask.
            x_simulated = _simulate_model(x, score_mask)
            # Scale the simulated instance.
            x_simulated = scaler.scale(x_simulated)

            # Compute the Spatial-Temporal GNN model predictions on the
            # simulated instance.
            y_pred = spatial_temporal_gnn(x_simulated.unsqueeze(0))

            # Un-scale the predictions.
            y_pred = scaler.un_scale(y_pred)

            mae = mae_criterion(y_pred, y.unsqueeze(0))
            rmse = rmse_criterion(y_pred, y.unsqueeze(0))
            mape = mape_criterion(y_pred, y.unsqueeze(0))

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
"""

from math import ceil
from time import time
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.spatial_temporal_gnn.model import SpatialTemporalGNN
from src.spatial_temporal_gnn.metrics import MAE, RMSE, MAPE
from src.spatial_temporal_gnn.training import Checkpoint
from src.explanation.navigator.model import Navigator
from src.data.data_processing import Scaler

def _simulate_model(
    instance: torch.FloatTensor, events_scores: torch.FloatTensor
    ) -> torch.FloatTensor:
    # Apply the sigmoid function to the events scores.
    events_scores = events_scores.sigmoid()
    # Get a random uniform tensor in order to apply the differentiable
    # relaxed Bernoulli function.
    eps = torch.rand_like(events_scores)
    # Compute the relaxed Bernoulli distribution and apply the sigmoid
    # function to the resulting scores.
    events_scores = torch.sigmoid((eps.log() - (1 - eps).log() + events_scores) / 2.0)
    # TODO: Simulate all events, not just the speed events.
    # Threshold the events scores on 0.5.
    result = events_scores >= .5
    # Mask the instance according to the thresholded event scores.
    instance = result * instance
    return instance

def train(
    model: Navigator, optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader, val_dataloader: DataLoader,
    spatial_temporal_gnn: SpatialTemporalGNN, scaler: Scaler,
    epochs: int, validations_per_batch: int = 1,
    checkpoint: Optional[Checkpoint] = None,
    lr_scheduler: Optional[object] = None,
    reload_best_weights: bool = True) -> Dict[str, np.ndarray]:
    # Get the device that is used for training and querying the model.
    device = model.device
    
    # Set the valdation step inside the batch
    assert validations_per_batch > 0, \
        'The number of validations per batch must be greater than zero.'
    val_step = ceil(len(train_dataloader) / validations_per_batch)

    # Initialize the training criterions.
    mae_criterion = MAE(missing_value=float('nan'))
    rmse_criterion = RMSE(missing_value=float('nan'))
    mape_criterion = MAPE(missing_value=float('nan'))

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

        for batch_idx, (x, target_event, events_scores) in enumerate(
            train_dataloader):
            # Increment the number of batch steps.
            batch_steps = batch_idx + 1

            x = x.float().to(device=device)
            target_event = target_event.float().to(device=device)

            # Repeat the target event for all the input events.
            #target_repeated = target_event.unsqueeze(0).repeat(
            #    x.shape[0], 1)

            # Compute the correlation scores between each input event and
            # the target event.
            predicted_event_scores = model([], x, target_event)

            # Compute the MAE loss.
            # Mask out but preserve gradients for the events that are not
            # present in the input instance.
            #predicted_event_scores = predicted_event_scores * (x != 0).float()
            #print(predicted_event_scores.shape, events_scores.shape)
            events_scores[x[..., 0] == 0.] = float('nan')
            events_scores = events_scores.unsqueeze(-1).to(device)
            loss = mae_criterion(predicted_event_scores, events_scores)

            # Compute the RMSE and MAPE metrics.
            with torch.no_grad():
                rmse = rmse_criterion(predicted_event_scores, events_scores)
                mape = mape_criterion(predicted_event_scores, events_scores)

            # Update the running errors.
            running_train_mae += loss.item()
            running_train_rmse += rmse.item()
            running_train_mape += mape.item()

            # Zero the gradients.
            optimizer.zero_grad()

            # Use MAE as the loss function for backpropagation.
            loss.backward()

            # Zero the gradients of the Spatial-Temporal GNN model.
            #for param in spatial_temporal_gnn.parameters():
            #    param.grad[:] = 0

            # Update the weights.
            optimizer.step()

            # Get the batch time.
            epoch_time = time() - start_time
            batch_time = epoch_time / batch_steps

            # Check if it is time to apply validation.
            apply_validation = (batch_idx + 1) % val_step == 0
            
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
                end='\r' if not apply_validation else '\n')

            # Evaluate on validation set.
            if apply_validation:
                # Set the model in eval mode.
                model.eval()

                # Remove unused tensors from gpu memory.
                torch.cuda.empty_cache()

                # Compute the validation scores.
                val_results = validate(model, val_dataloader, spatial_temporal_gnn,
                                    scaler)
                val_mae, val_rmse, val_mape = val_results

                # Remove unused tensors from gpu memory.
                torch.cuda.empty_cache()
                
                # Print the validation step results.
                print(
                    '\t'
                    f'val step -',

                    f'val: {{ MAE: {val_mae:.3g} -',
                    f'RMSE: {val_rmse:.3g} -',
                    f'MAPE: {val_mape * 100.:.3g}% }} -',

                    f'lr: {optimizer.param_groups[0]["lr"]:.3g} -',
                    f'weight decay: {optimizer.param_groups[0]["weight_decay"]}'
                    )

                # Save the checpoints.
                if checkpoint is not None:
                    err_sum = val_mae + val_rmse + val_mape
                    checkpoint.save_best(model, optimizer, err_sum)

                # Set the model in train mode.
                model.train()

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
        val_results = validate(model, val_dataloader, spatial_temporal_gnn,
                               scaler)
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
        #lr_scheduler.step(train_mae)

        # Set model in training mode.
        lr_scheduler.step()
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
    model: Navigator, val_dataloader: DataLoader,
    spatial_temporal_gnn: SpatialTemporalGNN, scaler: Scaler
    ) -> Tuple[float, float, float]:
    device = model.device
    torch.cuda.empty_cache()

    # Initialize the validation criterions.
    mae_criterion = MAE(missing_value=float('nan'))
    rmse_criterion = RMSE(missing_value=float('nan'))
    mape_criterion = MAPE(missing_value=float('nan'))

    # Inizialize running errors.
    running_val_mae = 0.
    running_val_rmse = 0.
    running_val_mape = 0.

    with torch.no_grad():
        for x, target_event, events_scores in val_dataloader:
            x = x.float().to(device=device)
            target_event = target_event.float().to(device=device)

            # Repeat the target event for all the input events.
            #target_repeated = target_event.unsqueeze(0).repeat(
            #    x.shape[0], 1)

            # Compute the correlation scores between each input event and
            # the target event.
            predicted_event_scores = model([], x, target_event)

            events_scores[x[..., 0] == 0.] = float('nan')
            events_scores = events_scores.unsqueeze(-1).to(device)
            mae = mae_criterion(predicted_event_scores, events_scores)
            rmse = rmse_criterion(predicted_event_scores, events_scores)
            mape = mape_criterion(predicted_event_scores, events_scores)

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