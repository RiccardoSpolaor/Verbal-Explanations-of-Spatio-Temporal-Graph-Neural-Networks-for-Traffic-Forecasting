import gc
import os
import pickle
import random
from copy import deepcopy
from tqdm.auto import tqdm
from typing import List, Dict, Union, Optional

from typing import Any, List

import torch

from ...spatial_temporal_gnn.model import SpatialTemporalGNN
from ...spatial_temporal_gnn.metrics import MAE
from ...data.data_processing import Scaler
from ...explanation.events import remove_features_by_events

class Game:
    def __init__(
        self,
        input_events: List[List[float]],
        leaf_size: int,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        spatial_temporal_gnn: SpatialTemporalGNN,
        scaler: Scaler
        ) -> None:
        self.input_events = input_events
        self.leaf_size = leaf_size
        self.x = x
        self.y = y
        self.spatial_temporal_gnn = spatial_temporal_gnn
        self.scaler = scaler
        self.mae = MAE()
        

    def get_state(self) -> Any:
        return hash(frozenset(self.input_events))
    
    def possible_actions(self) -> List[int]:
        return list(range(len(self.input_events)))

    def take_action(self, action: int) -> None:
        # Remove at index action
        self.input_events.pop(action)

    def has_outcome(self) -> bool:
        return len(self.input_events) <= self.leaf_size
    
    def reward(self) -> float:
        # Clone the input data to avoid modifying it.
        x = self.x.clone()
        # Set the MAE criterion.
        mae_criterion = MAE()
        # Get the device of the spatial temporal GNN.
        device = self.spatial_temporal_gnn.device

        # Set the input events as a list.
        input_events = [[0, e[0], e[1]] for e in self.input_events]
        # Remove the features not corresponding to the input events in
        # the input data.

        # Add the batch dimension to the data and add them to the
        # device.
        x_subset = x.unsqueeze(0).to(device)
        #x_subset = scaler.scale(x)
        x_subset = self.scaler.scale(x_subset)
        x_subset = remove_features_by_events(x_subset, input_events, remove_value=-2.25)
        
        y = self.y.unsqueeze(0).to(device)

        # Predict the output graph.
        y_pred = self.spatial_temporal_gnn(x_subset)

        # Scale the prediction.
        y_pred = self.scaler.un_scale(y_pred)
        # Remove the non-considered target features in the prediction.
        y_pred[y == 0] = 0

        # Compute the reward as the negative MAE between the predicted
        # output events and the actual output events.
        reward = - mae_criterion(y_pred, y).item()
        return reward

class Node:
    def __init__(self, state: int, prev_node: Optional['Node'] = None, transposition_table: Optional[Dict[tuple, str]] = None):
        self.state = state
        
        self.prev_node = prev_node
        self.transposition_table = transposition_table # {(player, state): Node}
        self.children = dict() # {action: Node}

        self.is_expanded = False
        self.has_outcome = False

        self.w = 0. # number of games won by previous player where node was traversed
        self.n = 0 # number of games played where node was traversed

    def eval(self) -> float:
        return self.w / self.n if self.n > 0 else float("inf")

    def add_child(self, next_state: str, action: int) -> None:
        if action not in self.children:
            if self.transposition_table is not None:
                key = next_state
                if key in self.transposition_table:
                    self.children[action] = self.transposition_table[key]
                else:
                    self.children[action] = self.transposition_table[key] = Node(next_state, transposition_table = self.transposition_table)
            else:
                self.children[action] = Node(next_state, prev_node = self)

    def choose_best_action(self) -> int:
        return max(self.children, key = lambda action: self.children[action].eval())

    def choose_random_action(self) -> int:
        return random.sample(self.children.keys(), 1)[0]

class MCTS:
    def __init__(
        self,
        input_events,
        leaf_size,
        x,
        y,
        spatial_temporal_gnn,
        scaler,
        allow_transpositions: bool = True):
        self.game = Game(input_events, leaf_size, x, y, spatial_temporal_gnn, scaler)
        self.copied_game = deepcopy(self.game)

        self.transposition_table = dict() if allow_transpositions is True else None
        self.root = Node(str(self.game.get_state()), transposition_table = self.transposition_table)
        if self.transposition_table is not None:
            self.transposition_table[self.game.get_state()] = self.root
        
    def selection(self, node: Node) -> List[Node]:
        path = [node]
        while path[-1].is_expanded is True and path[-1].has_outcome is False: # loop if not leaf node
            action = path[-1].choose_best_action()
            path.append(path[-1].children[action])
            self.copied_game.take_action(action)
        return path

    def expansion(self, path: List[Node]) -> List[Node]:
        if path[-1].is_expanded is False and path[-1].has_outcome is False:
            for action in self.copied_game.possible_actions():
                expanded_game = deepcopy(self.copied_game)
                expanded_game.take_action(action)
                path[-1].add_child(expanded_game.get_state(), action)

            assert len(path[-1].children) > 0
            
            path[-1].is_expanded = True
            action = path[-1].choose_random_action()
            path.append(path[-1].children[action])
            self.copied_game.take_action(action)
        return path

    def simulation(self, path: List[Node]) -> List[Node]:
        while self.copied_game.has_outcome() is False:
            action = random.choice(self.copied_game.possible_actions())
            self.copied_game.take_action(action)
            path[-1].add_child(self.copied_game.get_state(), action)
            path.append(path[-1].children[action])
        return path

    def backpropagation(self, path: List[Node]) -> None:
        if self.copied_game.has_outcome() is True:
            reward = self.copied_game.reward()
            print(reward)
            path[0].n += 1
            for i in range(1, len(path)):
                path[i].w += - reward
                path[i].n += 1
            path[-1].has_outcome = True

    def step(self) -> None:
        self.backpropagation(self.simulation(self.expansion(self.selection(self.root))))

        self.copied_game = deepcopy(self.game)
        gc.collect()

    def self_play(self, iterations: int = 1) -> None:
        for _ in tqdm(range(iterations), 'Playing...'):
            self.step()