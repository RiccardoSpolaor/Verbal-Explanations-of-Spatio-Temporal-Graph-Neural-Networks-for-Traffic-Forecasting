from time import time
from typing import List, Tuple

from sklearn.model_selection import ParameterGrid
import torch
import numpy as np

from .search import get_best_input_events_subset_by_mcts
from ..events import remove_features_by_events
from ...spatial_temporal_gnn.model import SpatialTemporalGNN
from ...spatial_temporal_gnn.metrics import MAE, RMSE, MAPE
from ...data.data_processing import Scaler
from ...utils.config import (
    SEVERE_CONGESTION_THRESHOLD_MPH,
    CONGESTION_THRESHOLD_MPH)


def get_all_explanations(
    x: np.ndarray,
    y: np.ndarray,
    distance_matrix: np.ndarray,
    spatial_temporal_gnn: SpatialTemporalGNN,
    scaler: Scaler,
    n_rollouts: int = 30,
    explanation_size_factor: int = 3,
    cut_size_factor: int = 2,
    exploration_weight: float = 20,
    remove_value: float = 0.,
    divide_by_traffic_cluster_kind: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    explained_x, explained_y, scores = [], [], []

    mae_criterion = MAE()
    rmse_criterion = RMSE()
    mape_criterion = MAPE()

    if divide_by_traffic_cluster_kind:
        running_mae_severe_congestion = None
        running_mae_congestion = None
        running_mae_free_flow = None
        running_rmse_severe_congestion = None
        running_rmse_congestion = None
        running_rmse_free_flow = None
        running_mape_severe_congestion = None
        running_mape_congestion = None
        running_mape_free_flow = None

        severe_congestions_steps = 0
        congestions_steps = 0
        free_flows_steps = 0

    running_mae = 0.
    running_rmse = 0.
    running_mape = 0.
    running_time = 0.

    for i, (x_, y_) in enumerate(zip(x, y)):
        steps = i + 1

        start_time = time()

        x_subset, input_events_subset = get_instance_explanations(
            x_,
            y_,
            distance_matrix,
            spatial_temporal_gnn,
            scaler,
            n_rollouts=n_rollouts,
            explanation_size_factor=explanation_size_factor,
            cut_size_factor=cut_size_factor,
            exploration_weight=exploration_weight,
            remove_value=remove_value,
            verbose=False)

        mae, rmse, mape = evaluate(
            x_,
            y_,
            input_events_subset,
            spatial_temporal_gnn,
            scaler,
            mae_criterion,
            rmse_criterion,
            mape_criterion,
            remove_value=remove_value)

        running_time += time() - start_time
        running_mae += mae
        running_rmse += rmse
        running_mape += mape

        if divide_by_traffic_cluster_kind:
            if np.all(y_[y_.nonzero()] <= SEVERE_CONGESTION_THRESHOLD_MPH):
                if severe_congestions_steps == 0:
                    running_mae_severe_congestion = mae
                    running_rmse_severe_congestion = rmse
                    running_mape_severe_congestion = mape
                running_mae_severe_congestion += mae
                running_rmse_severe_congestion += rmse
                running_mape_severe_congestion += mape

                severe_congestions_steps += 1
            elif np.all(y_[y_.nonzero()] <= CONGESTION_THRESHOLD_MPH):
                if running_mae_congestion is None:
                    running_mae_congestion = mae
                    running_rmse_congestion = rmse
                    running_mape_congestion = mape
                running_mae_congestion += mae
                running_rmse_congestion += rmse
                running_mape_congestion += mape

                congestions_steps += 1
            else:
                if running_mae_free_flow is None:
                    running_mae_free_flow = mae
                    running_rmse_free_flow = rmse
                    running_mape_free_flow = mape
                running_mae_free_flow += mae
                running_rmse_free_flow += rmse
                running_mape_free_flow += mape

                free_flows_steps += 1
            
            mae_severe_congestion_avg = 'N/D' if severe_congestions_steps == 0 \
                else f'{running_mae_severe_congestion / severe_congestions_steps:.3g}'
            rmse_severe_congestion_avg = 'N/D' if severe_congestions_steps == 0 \
                else f'{running_rmse_severe_congestion / severe_congestions_steps:.3g}'
            mape_severe_congestion_avg = 'N/D' if severe_congestions_steps == 0 \
                else f'{running_mape_severe_congestion * 100. / severe_congestions_steps:.3g}'
                
            mae_congestion_avg = 'N/D' if congestions_steps == 0 \
                else f'{running_mae_congestion / congestions_steps:.3g}'
            rmse_congestion_avg = 'N/D' if congestions_steps == 0 \
                else f'{running_rmse_congestion / congestions_steps:.3g}'
            mape_congestion_avg = 'N/D' if congestions_steps == 0 \
                else f'{running_mape_congestion * 100. / congestions_steps:.3g}'
            
            mae_free_flow_avg = 'N/D' if free_flows_steps == 0 \
                else f'{running_mae_free_flow / free_flows_steps:.3g}'
            rmse_free_flow_avg = 'N/D' if free_flows_steps == 0 \
                else f'{running_rmse_free_flow / free_flows_steps:.3g}'
            mape_free_flow_avg = 'N/D' if free_flows_steps == 0 \
                else f'{running_mape_free_flow * 100. / free_flows_steps:.3g}'

            print(
                f'[{steps}/{len(x)}] -',
                f'{running_time:.0f}s -',

                f'MAE: {{ severe_congestion {mae_severe_congestion_avg} -'
                f'congestion {mae_congestion_avg} -'
                f'free_flow {mae_free_flow_avg} -',
                f'total: {running_mae / steps:.3g} }} -',
                f'RMSE: {{ severe_congestion {rmse_severe_congestion_avg} -'
                f'congestion {rmse_congestion_avg} -'
                f'free_flow {rmse_free_flow_avg} -',
                f'total: {running_rmse / steps:.3g} }} -',
                f'MAPE: {{ severe_congestion {mape_severe_congestion_avg}% -'
                f'congestion {mape_congestion_avg}% -'
                f'free_flow {mape_free_flow_avg}% -',
                f'total: {running_mape * 100. / steps:.3g}% }} -',
                f'Average time: {running_time / steps:.3g}s',

                '             ' if steps < len(x) else '',
                end='\r' if steps < len(x) else '\n'
                )
        else:
            print(
            f'[{steps}/{len(x)}] -',
            f'{running_time:.0f}s -',

            f'MAE: {running_mae / steps:.3g} -',
            f'RMSE: {running_rmse / steps:.3g} -',
            f'MAPE: {running_mape * 100. / steps:.3g}% -',
            f'Average time: {running_time / steps:.3g}s',

            '             ' if steps < len(x) else '',
            end='\r' if steps < len(x) else '\n'
            )

        explained_x.append(x_subset)
        explained_y.append(y_)
        scores.append((mae, rmse, mape))

    return np.array(explained_x), np.array(explained_y), np.array(scores)

def get_instance_explanations(
    x: np.ndarray,
    y: np.ndarray,
    distance_matrix: np.ndarray,
    spatial_temporal_gnn: SpatialTemporalGNN,
    scaler: Scaler,
    n_rollouts: int = 30,
    explanation_size_factor: int = 3,
    cut_size_factor: int = 2,
    exploration_weight: float = 20,
    remove_value: float = 0.,
    verbose: bool = False
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    explanation_size = (y.flatten() != 0).sum() * explanation_size_factor
    n_top_events = explanation_size * cut_size_factor

    events_subset = get_best_input_events_subset_by_mcts(
        x,
        y,
        distance_matrix,
        spatial_temporal_gnn,
        scaler,
        n_rollouts=n_rollouts,
        explanation_size=explanation_size,
        n_top_events=n_top_events,
        exploration_weight=exploration_weight,
        remove_value=remove_value,
        verbose=verbose,
        )

    x_subset = remove_features_by_events(
        x,
        events_subset,
        remove_value=0.)

    return x_subset, events_subset

def apply_grid_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    distance_matrix: np.ndarray,
    spatial_temporal_gnn: SpatialTemporalGNN,
    scaler: Scaler,
    n_rollouts_list: List[int],
    explanation_size_factor_list: List[int],
    cut_size_factor_list: List[int],
    exploration_weight_list: List[float],
    remove_value_list: float = 0.
    ) -> None:
    """
    Apply grid search to find the best hyperparameters for the Monte Carlo
    algorithm with respect to the given dataset.

    Parameters
    ----------
    x_train : ndarray
        The input dataset used to evaluate the Monte Carlo algorithm.
    y_train : np.ndarray
        The target dataset used to evaluate the Monte Carlo algorithm.
    distance_matrix : ndarray
        The matrix containing the harvesine distance between each pair of
        nodes in miles.
    spatial_temporal_gnn : SpatialTemporalGNN
        The spatial-temporal graph neural network used to predict the target
        values.
    scaler : Scaler
        The scaler used to scale and un-scale the input and target datasets.
    n_rollouts_list : list of int
        List of values to be tested for the number of rollouts
        hyperparameter.
    explanation_size_factor_list : list of int
        List of values to be tested for the explanation size factor
        hyperparameter.
    cut_size_factor_list : list of int
        List of values to be tested for the cut size factor
        hyperparameter.
    exploration_weight_list : lidt of float
        List of values to be tested for the exploration weight
        hyperparameter.
    """
    parameter_grid = ParameterGrid({
        'n_rollouts': n_rollouts_list,
        'explanation_size_factor': explanation_size_factor_list,
        'cut_size_factor': cut_size_factor_list,
        'exploration_weight': exploration_weight_list,
        'remove_value': remove_value_list
        })

    for p in parameter_grid:
        print('Testing:', *[f'{k}: {v}' for k, v in p.items()])
        get_all_explanations(
            x_train,
            y_train,
            distance_matrix,
            spatial_temporal_gnn,
            scaler,
            divide_by_traffic_cluster_kind=True,
            **p)
        print()

def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    input_events_subset: List[Tuple[int, int]],
    spatial_temporal_gnn: SpatialTemporalGNN,
    scaler: Scaler,
    mae_criterion: MAE,
    rmse_criterion: RMSE,
    mape_criterion: MAPE,
    remove_value: float = 0.
    ) -> Tuple[float, float, float]:
    """
    Evaluate the resulting input events subset obtained by the Monte Carlo
    algorithm as an explanation for the given target events.
    The output spatial-temporal graph subset by the target events is
    re-predicted by the input spatial-temporal graph considering just the
    input events subset.
    The evaluation is performed by computing the Mean Absolute Error (MAE),
    the Root Mean Squared Error (RMSE), and the Mean Absolute Percentage
    Error (MAPE) between the original output spatial-temporal graph subset
    and re-predicted one.

    Parameters
    ----------
    x : ndarray
        The input spatial-temporal graph.
    y : ndarray
        The target spatial-temporal graph subset by the target events.
    input_events_subset : list of (int, int)
        The input events subset obtained by the Monte Carlo algorithm
        as an explanation for the given target events.
    spatial_temporal_gnn : SpatialTemporalGNN
        The spatial-temporal graph neural network used to predict the target
        values.
    scaler : Scaler
        The scaler used to scale and un-scale the input and target datasets.
    mae_criterion : MAE
        The MAE criterion.
    rmse_criterion : RMSE
        The RMSE criterion.
    mape_criterion : MAPE
        The MAPE criterion.
    remove_value : float, optional
        The value used to remove the speed features not related to the input
        events subset in the input spatial-temporal graph, by default 0.

    Returns
    -------
    float
        The MAE between the original output spatial-temporal graph subset
        and re-predicted one.
    float
        The RMSE between the original output spatial-temporal graph subset
        and re-predicted one.
    float
        The MAPE between the original output spatial-temporal graph subset
        and re-predicted one.
    """
    with torch.no_grad():
        x = torch.FloatTensor(x).to(spatial_temporal_gnn.device)
        x_subset = remove_features_by_events(
            x,
            input_events_subset,
            remove_value)
        x_subset = scaler.scale(x_subset)
        x_subset = x_subset.unsqueeze(0)

        y = torch.FloatTensor(y).to(spatial_temporal_gnn.device)
        y = y.unsqueeze(0)

        y_pred = spatial_temporal_gnn(x_subset)
        y_pred = scaler.un_scale(y_pred)

        mae = mae_criterion(y_pred, y)
        rmse = rmse_criterion(y_pred, y)
        mape = mape_criterion(y_pred, y)

        return mae.item(), rmse.item(), mape.item()

def print_scores_report(scores: np.ndarray, dataset_name: str) -> None:
    len_scores = len(scores)
    traffic_cluster_kinds = [
        'Severe Congestions',
        'Moderate Congestions',
        'Free Flows']

    instances_per_traffic_cluster_kind = len_scores // 3
    print(f'Scores report for the {dataset_name} dataset:')
    for i, k in enumerate(traffic_cluster_kinds):
        start = i * instances_per_traffic_cluster_kind
        end = (i + 1) * instances_per_traffic_cluster_kind
        print(
            f'\t{k} scores :',
            f'MAE: {scores[start:end, 0].mean():.3g} -',
            f'RMSE: {scores[start:end, 1].mean():.3g} -',
            f'MAPE: {scores[start:end, 2].mean() * 100.:.3g}%')
