"""
Module containing the Monte Carlo Tree Search (MCTS) class
and its Node class.
"""
import math
from copy import deepcopy
from collections import defaultdict
from typing import List, Literal, Tuple, Union

import torch
import numpy as np

from ..events import remove_features_by_events
from ...spatial_temporal_gnn.metrics import MAE
from ...spatial_temporal_gnn.model import SpatialTemporalGNN
from ...data.data_processing import Scaler


class Node():
    """
    Class representing a node in the Monte Carlo Tree.
    A node is described by a list of input events. The reward of a node
    is the reciprocal Mean Absolute Error (MAE) between the predicted
    output data, given the subset of input events expressed by the node,
    and the actual output data.
    
    Attributes
    ----------
    input_events : list of (int, int)
        The input events expressed by the node.
    """
    def __init__(self, input_events: List[Tuple[int, int]]) -> None:
        """
        Initialize the node.

        Parameters
        ----------
        input_events : list of (int, int)
            The input events expressed by the node.
        """
        self.input_events = input_events

    def is_terminal(self, leaf_size: int) -> bool:
        """
        Returns True if the node has less than or equal to `leaf_size`
        events.

        Parameters
        ----------
        leaf_size : int
            The maximum number of events allowed in a leaf node.

        Returns
        -------
        bool
            Whether or not the node is terminal.
        """
        return len(self.input_events) <= leaf_size

    def reward(
        self, 
        x: torch.FloatTensor, 
        y: torch.FloatTensor,
        spatial_temporal_gnn: SpatialTemporalGNN,
        scaler: Scaler,
        mae_criterion: MAE,
        remove_value: Union[float, Literal['perturb']] = 0.
        ) -> float:
        """
        Get the reward of the current node in terms of the reciprocal
        Mean Absolute Error (MAE) between the predicted output data,
        given the subset of input events expressed by the current node,
        and the actual output data.

        Parameters
        ----------
        x : FloatTensor
            The input data.
        y : FloatTensor
            The output data.
        spatial_temporal_gnn : SpatialTemporalGNN
            The Spatial-Temporal GNN model used to predict the output
            events.
        scaler : Scaler
            The scaler used to scale and un-scale the data.
        mae_criterion : MAE
            The MAE criterion used to compute the reward.
        remove_value : float | perturb, optional
            The value used to substitute the speed features of the 
            input events not present in the description of the node.
            If 'perturb', the value is sampled from a normal distribution.
            By default 0.

        Returns
        -------
        float
            The reciprocal MAE between the predicted output data and the
            actual output data.
        """
        with torch.no_grad():
            # Clone the input data to avoid modifying it.
            x = x.clone()
            # Get the device of the spatial temporal GNN.
            device = spatial_temporal_gnn.device

            # Get the subset of the input data corresponding to the input
            # events of the node.
            x_subset = remove_features_by_events(
                x,
                self.input_events,
                remove_value=remove_value)
            # Scale the input data.
            x_subset = scaler.scale(x_subset)
            # Add the batch dimension to the subset.
            x_subset = x_subset.unsqueeze(0)
            # Move the subset to the device.
            x_subset = x_subset.to(device)

            # Add the batch dimension to the output data.
            y = y.unsqueeze(0).to(device)

            # Predict the output graph given the subset of the input data.
            y_pred = spatial_temporal_gnn(x_subset)

            # Scale the prediction.
            y_pred = scaler.un_scale(y_pred)
            # Remove the non-considered target features in the prediction.
            y_pred[y == 0] = 0

            # Compute the reward as the reciprocal MAE between the predicted
            # output events and the actual output events.
            mae = mae_criterion(y_pred, y).item()
            if mae == 0.:
                return float('inf'), mae
            return 1. / mae, mae

    def __hash__(self) -> int:
        """Hash the node by the input events that describe it.

        Returns
        -------
        int
            The hash of the node.
        """
        return hash(frozenset(self.input_events))

    def __eq__(self, node2: 'Node') -> bool:
        """Get whether or not two nodes are equal.
        A node is equal to another node if they have the same input
        events set.

        Parameters
        ----------
        node1 : Node
            The first node to compare.
        node2 : Node
            The second node to compare.

        Returns
        -------
        bool
            Whether or not the two nodes are equal.
        """
        return frozenset(self.input_events) == frozenset(node2.input_events)

class MonteCarloTreeSearch:
    """
    Class representing the Monte Carlo Tree Search (MCTS) algorithm.
    It is used to find the best subset of input events to explain the
    output events.
    
    Attributes
    ----------
    x : FloatTensor
        The input data.
    y : FloatTensor
        The output data.
    spatial_temporal_gnn : SpatialTemporalGNN
        The Spatial Temporal Graph Neural Network used to get the
        predictions from a leaf node.
    scaler : Scaler
        The scaler used to scale and un-scale the data.
    maximum_leaf_size : int
        The maximum number of events allowed in a leaf node.
    exploration_weight : int
        The exploration weight used in the Upper Confidence Bound for
        Trees (UCT) formula.
    remove_value : float | 'perturb'
        The value used to substitute the speed features of the input
        events not present in the description of the node.
        If 'perturb', the value is sampled from a normal distribution.
    mae_criterion : MAE
        The MAE criterion used to compute the reward.
    C : DefaultDict
        The dictionary of total reward of each node.
    N : DefaultDict
        The dictionary of total visit count for each node.
    branching_actions : Dict
        The dictionary of possible branching actions for each node,
        that define how to expand them. A branching action is an
        event of the input events set that can be removed to expand 
        a child node.
    expanded_children : Dict
        The dictionary of expanded children of each node.
    best_leaf : (Node, float)
        The best leaf node found by the MCTS so far and its reward.
    """

    def __init__(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        spatial_temporal_gnn: SpatialTemporalGNN,
        scaler: Scaler,
        maximum_leaf_size: int = 20,
        exploration_weight: int = 1,
        remove_value: Union[float, Literal['perturb']] = 0.
        ) -> None:
        """
        Initialize the MCTS.

        Parameters
        ----------
        x : FloatTensor
            The input data.
        y : FloatTensor
            The output data.
        spatial_temporal_gnn : SpatialTemporalGNN
            The Spatial Temporal Graph Neural Network used to get the
            predictions from a leaf node.
        scaler : Scaler
            The scaler used to scale and un-scale the data.
        maximum_leaf_size : int, optional
            The maximum number of events allowed in a leaf node, by
            default 20.
        exploration_weight : int, optional
            The exploration weight used in the Upper Confidence Bound
            for Trees (UCT) formula, by default 1.
        remove_value : float | 'perturb', optional
            The value used to substitute the speed features of the
            input events not present in the description of the node.
            If 'perturb', the value is sampled from a normal distribution.
            By default 0.
        """
        # Set the inputs.
        self.x = x.clone()
        # Set the outputs.
        self.y = y.clone()
        # Set the Spatial Temporal Graph Neural Network.
        self.spatial_temporal_gnn = spatial_temporal_gnn
        # Set the scaler.
        self.scaler = scaler
        # Set the maximum leaf size.
        self.maximum_leaf_size = maximum_leaf_size
        # Set the exploration weight.
        self.exploration_weight = exploration_weight
        # Set the remove value.
        self.remove_value = remove_value
        # Set the MAE criterion.
        self.mae_criterion = MAE()

        # Set dictionary of total reward of each node.
        self.C = defaultdict(int)
        # Set dictionary of total visit count for each node.
        self.N = defaultdict(int)
        # Set dictionary of possible branching actions of each node.
        self.branching_actions = dict()
        # Set dictionary of expanded children of each node.
        self.expanded_children = dict()
        # Set the best found leaf node along with its error.
        self.best_leaf = ( None, - math.inf, math.inf )

    def rollout(self, node: Node) -> None:
        """
        Do a Monte Carlo Tree Search rollout starting from the given
        root node and reaching a leaf node. After the rollout, the leaf node
        is saved as the best leaf node if it has a lower error than the
        current best leaf node. Moreover, the reward is backpropogated
        from the leaf node to the root node in order to update the
        total reward and total visit count of each node.

        Parameters
        ----------
        node : Node
            The root node of the tree search.
        """
        # Get the path from the root node to the leaf node.
        # Apply random node expansion, and then node selection through
        # the Upper Confidence Bound applied to Trees (UCT) formula.
        path = self._select(node)
        # Get the leaf node.
        leaf = path[-1]
        # Get the reward of the leaf node.
        reward = self._simulate(leaf)
        # Backpropogate the reward from the leaf node to the root node.
        self._backpropagate(path, reward)

    def get_best_input_events_subset(self) -> List[Tuple[int, int]]:
        """
        Get the best subset of input events found by the MCTS.

        Returns
        -------
        list of (int, int)
            The best subset of input events found by the MCTS.
        """
        return self.best_leaf[0].input_events

    def _select(self, node: Node) -> List[Node]:
        """
        Select a leaf node by applying node expansion through the
        navigator and node selection through the Upper Confidence Bound
        applied to Trees (UCT). Return the path from the root node to
        the leaf node.
        
        Parameters
        ----------
        node : Node
            The root node of the tree search.
        
        Returns
        -------
        list of Node
            The path from the root node to the leaf node.
        """
        # Set the rollout path.
        path = []
        while True:
            # Append the node to the path.
            path.append(node)
            # If the node is a terminal node, return the path.
            if node.is_terminal(self.maximum_leaf_size):
                return path
            # Expand the node children.
            self._expand(node)
            # Explore the child node that maximizes the Upper Confidence
            # Bound applied to Trees (UCT) formula.
            node = self._get_node_by_upper_confidence_bound(node)

    def _expand(self, node: Node) -> None:
        """
        Expand the children of the given node.
        
        Parameters
        ----------
        node : Node
            The node to expand.
        """
        if node not in self.branching_actions.keys():
            # The node has never been visited yet.
            if node not in self.expanded_children.keys():
                # Initialize the children of the node with all possible
                # branches (all possible input events).
                self.branching_actions[node] = deepcopy(node.input_events)
                # Initialize the expanded children of the node with an
                # empty list (no children have been expanded).
                self.expanded_children[node] = []
            # The node has been fully expanded.
            else:
                return
        # Set the children input events.
        children_events = deepcopy(node.input_events)
        if len(self.branching_actions[node]):
            # Get a random index of the branch to expand (event to remove).
            index = np.random.choice(len(self.branching_actions[node]))
            # Remove the selected event from the events of the children.
            children_events.remove(self.branching_actions[node][index])
            # Delete the expanded branch from the branches of the node.
            del self.branching_actions[node][index]
        # Expand the node in accordance to the removed branch, by
        # removing the event corresponding to the branch.
        self.expanded_children[node].append(Node(children_events))

    def _simulate(self, node: Node) -> float:
        """
        Simulate the reward of the given leaf node through the Spatial
        Temporal Graph Neural Network. If the reward is higher than the
        reward of the current best leaf node, then the leaf node is
        saved as the best leaf node.
        
        Parameters
        ----------
        node : Node
            The leaf node to simulate.
            
        Returns
        -------
        float
            The reward of the leaf node.
        """
        reward, mae = node.reward(
            self.x,
            self.y,
            self.spatial_temporal_gnn,
            self.scaler,
            self.mae_criterion,
            remove_value=self.remove_value)

        if reward > self.best_leaf[1]:
            self.best_leaf = (node, reward, mae)

        return reward

    def _backpropagate(self, path: List[Node], reward: float) -> None:
        """
        Backpropagate the reward from the node to its ancestors.

        Parameters
        ----------
        path : list of Node
            The path from the root node to the leaf node.
        reward : float
            The reward score of the leaf node to backpropagate.
        """
        for node in reversed(path):
            # Update the total visit count of the node.
            self.N[node] += 1
            # Update the total reward of the node.
            self.C[node] += reward
            # Update the reward.
            # reward += 1

    def _get_node_by_upper_confidence_bound(self, node: Node) -> Node:
        """
        Get a child node of the given node by the Upper Confidence Bound
        for Trees (UCT) algorithm, balancing exploration & exploitation.

        Parameters
        ----------
        node : Node
            The parent node to get a child of to explore.

        Returns
        -------
        Node
            The child node to explore.
        """
        # Get the sum of total visit count of each children.
        N_sum = sum([self.N[c] for c in self.expanded_children[node]])

        def get_upper_confidence_bound(n: Node) -> float:
            """
            Get the Upper Confidence Bound for Trees (UCT) of a child
            Node.

            Parameters
            ----------
            n : Node
                The child node to get the UCT of.

            Returns
            -------
            float
                The UCT of the child node.
            """
            if self.N[n] == 0:
                return float('inf')
            return self.C[n] / (self.N[n]) + self.exploration_weight / self.best_leaf[1] *\
                math.sqrt(N_sum) / (self.N[n])

        return max(self.expanded_children[node], key=get_upper_confidence_bound)
