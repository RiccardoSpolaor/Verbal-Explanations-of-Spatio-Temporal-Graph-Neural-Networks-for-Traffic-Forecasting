from typing import List
from collections import defaultdict
import math
import torch
import numpy as np

from src.spatial_temporal_gnn.metrics import MAE
from src.spatial_temporal_gnn.model import SpatialTemporalGNN
from src.data.data_processing import Scaler
from src.explanation.navigator.model import Navigator
from src.explanation.events import (
    remove_features_by_events, get_largest_event_set)


class Node():
    """
    Class representing a node in the Monte Carlo Tree Search.
    A node is a subset of input events. The children of a node are
    the subsets of input events obtained by removing one event from
    the node. The reward of a node is the negative Mean Absolute Error
    (MAE) between the predicted output data, given the subset of input
    events expressed by the node, and the actual output data.
    
    Attributes
    ----------
    input_events : list of list of float
        The input events expressed by the node.
    """
    def __init__(self, input_events: List[List[float]]) -> None:
        """Initialize the node.

        Parameters
        ----------
        input_events : list of list of float
            The input events expressed by the node.
        """
        self.input_events = input_events

    def find_children(self) -> List[List[int]]:
        """
        Get all possible successors of the current node.

        Returns
        -------
        set of Node
            All possible successors of the current node.
        """
        children = []

        for i, _ in enumerate(self.input_events):
            input_events_subset = self.input_events[:i] + self.input_events[i+1:]
            children.append(Node([ e for e in input_events_subset ]))

        return children

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
        self, spatial_temporal_gnn: SpatialTemporalGNN,
        x: torch.FloatTensor, y: torch.FloatTensor,
        scaler: Scaler) -> float:
        """
        Get the reward of the current node in terms of the negative
        Mean Absolute Error (MAE) between the predicted output data,
        given the subset of input events expressed by the current node,
        and the actual output data.

        Parameters
        ----------
        spatial_temporal_gnn : SpatialTemporalGNN
            The Spatial Temporal GNN model used to predict the output
            events.
        x : FloatTensor
            The input data.
        y : FloatTensor
            The output data.
        scaler : Scaler
            The scaler used to scale and un-scale the data.

        Returns
        -------
        float
            The negative MAE between the predicted output data and the
            actual output data.
        """
        # Clone the input data to avoid modifying it.
        x = x.clone()
        # Set the MAE criterion.
        mae_criterion = MAE()
        # Get the device of the spatial temporal GNN.
        device = spatial_temporal_gnn.device

        # Set the input events as a list.
        input_events = [[0, e[0], e[1]] for e in self.input_events]
        # Remove the features not corresponding to the input events in
        # the input data.

        # Add the batch dimension to the data and add them to the
        # device.
        x_subset = x.unsqueeze(0).to(device)
        #x_subset = scaler.scale(x)
        x_subset = scaler.scale(x_subset)
        x_subset = remove_features_by_events(x_subset, input_events, remove_value=-1)#-2.25)
        
        y = y.unsqueeze(0).to(device)

        # Predict the output graph.
        y_pred = spatial_temporal_gnn(x_subset)

        # Scale the prediction.
        y_pred = scaler.un_scale(y_pred)
        # Remove the non-considered target features in the prediction.
        y_pred[y == 0] = 0

        # Compute the reward as the negative MAE between the predicted
        # output events and the actual output events.
        reward = - mae_criterion(y_pred, y).item()
        return reward

    def __hash__(self) -> int:
        """Hash the node by the input events.

        Returns
        -------
        int
            The hash of the node.
        """
        return hash(frozenset(self.input_events))

    def __eq__(node1: 'Node', node2: 'Node') -> bool:
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
        return frozenset(node1.input_events) == frozenset(node2.input_events)
        #return node1.input_events == node2.input_events

class MonteCarloTreeSearch:
    """
    Class representing the Monte Carlo Tree Search (MCTS) algorithm.
    It is used to find the best subset of input events to explain the
    output events.
    
    Attributes
    ----------
    C : DefaultDict
        The dictionary of total reward of each node.
    N : DefaultDict
        The dictionary of total visit count for each node.
    children : Dict
        The dictionary of events that can be removed from a node to
        obtain a child node.
    expanded_children : Dict
        The dictionary of expanded children of each node.
    best_leaf : (Node, float)
        The best leaf node found by the MCTS so far and its reward.
    exploration_weight : int
        The exploration weight used in the Upper Confidence Bound for
        Trees (UCT) formula.
    spatial_temporal_gnn : SpatialTemporalGNN
        The Spatial Temporal Graph Neural Network used to get the
        reward of a leaf node.
    navigator : Navigator
        The Navigator used to select which node to expand during the
        tree search.
    scaler : Scaler
        The scaler used to scale and un-scale the data.
    x : FloatTensor
        The input data.
    y : FloatTensor
        The output data.
    maximum_leaf_size : int
        The maximum number of events allowed in a leaf node.
    """

    def __init__(
        self,
        spatial_temporal_gnn: SpatialTemporalGNN,
        scaler: Scaler,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        maximum_leaf_size: int = 20,
        exploration_weight: int = 1
        ) -> None:
        """Initialize the MCTS.

        Parameters
        ----------
        spatial_temporal_gnn : SpatialTemporalGNN
            The Spatial Temporal Graph Neural Network used to get the
            reward of a leaf node.
        navigator : Navigator
            The Navigator used to select which node to expand during
            the tree search.
        scaler : Scaler
            The scaler used to scale and un-scale the data.
        x : FloatTensor
            The input data.
        y : FloatTensor
            The output data.
        maximum_leaf_size : int, optional
            The maximum number of events allowed in a leaf node, by
            default 20.
        exploration_weight : int, optional
            The exploration weight used in the Upper Confidence Bound
            for Trees (UCT) formula, by default 1.
        """
        # Set dictionary of total reward of each node.
        self.C = defaultdict(int)
        # Set dictionary of total visit count for each node.
        self.N = defaultdict(int)
        # Set dictionary of children of each node.
        self.children = dict()
        # Set dictionary of expanded children of each node.
        self.expanded_children = dict()
        # Set the best found leaf node along with its error.
        self.best_leaf = ( None, - math.inf )
        # Set the exploration weight.
        self.exploration_weight = exploration_weight
        # Set the Spatial Temporal Graph Neural Network.
        self.spatial_temporal_gnn = spatial_temporal_gnn
        # Set the scaler.
        self.scaler = scaler
        # Set the maximum leaf size.
        self.maximum_leaf_size = maximum_leaf_size
        # Set the inputs.
        self.x = x.clone()
        # Set the outputs.
        self.y = y.clone()

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
        # Apply node expansion through the navigator and node selection
        # through the Upper Confidence Bound applied to Trees (UCT).
        path = self._select(node)
        # Get the leaf node.
        leaf = path[-1]
        # Get the reward of the leaf node.
        reward = self._simulate(leaf)
        # Backpropogate the reward from the leaf node to the root node.
        self._backpropagate(path, reward)
        #print(path)

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
        if node not in self.children.keys():
            if node not in self.expanded_children.keys():
                # The node has never been expanded yet.
                self.children[node] = [ e for e in node.input_events ]
                self.expanded_children[node] = []
            else:
                # The node has been fully expanded.
                return
        # Get the node input events.
        input_events = [ e for e in node.input_events ]
        # Remove the first child from the children of the node.
        if len(self.children[node]):
            # Choose with increasing probability over the length of the
            # children list the child to remove, the probability should sum up to 1.
            #probs = np.linspace(0, 1., len(self.children[node]))[::-1]
            # Get the index of the child to remove.
            index = np.random.choice(len(self.children[node]))#, p=probs)
            #index=0
            input_events.remove(self.children[node][index])
            # Delete the expanded child from the children of the node.
            del self.children[node][index]
        # Expand the node in accordance to the removed child.
        self.expanded_children[node].append(Node(input_events))

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
        reward = node.reward(
            self.spatial_temporal_gnn, self.x, self.y, self.scaler)

        if reward > self.best_leaf[1]:
            self.best_leaf = (node, reward)

        return reward

    def _backpropagate(self, path: List[Node], reward: float) -> None:
        """Backpropagate the reward from the node to its ancestors.

        Parameters
        ----------
        path : list of Node
            The path from the root node to the leaf node.
        reward : float
            The reward score of the leaf node to backpropagate.
        """
        # The reward is normalized by dividing it by 100.
        reward /= 100.
        for node in reversed(path):
            # Update the total visit count of the node.
            self.N[node] += 1
            # Update the total reward of the node.
            self.C[node] = max(self.C[node], reward)
            # Update the reward.
            reward += 1

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
                return float("inf")
            return self.C[n] / (self.N[n]) + self.exploration_weight *\
                math.sqrt(N_sum) / (self.N[n])

        return max(self.expanded_children[node], key=get_upper_confidence_bound)
