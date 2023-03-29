import torch
from torch import nn
import numpy as np

from .modules import S_GNN, GRU, Transformer, PositionalEncoder


class SpatialTemporalGNN(nn.Module):
    def __init__(self, n_in_features: int, n_out_features: int,
                 n_in_timesteps: int, n_out_timesteps: int, A: np.ndarray,
                 device: str, n_attention_heads: int = 4,
                 n_hidden_features: int = 64) -> None:
        super().__init__()
        # Set the number of hidden features.
        self.n_hidden_features = n_hidden_features

        # Get the refined adjacency matrix.
        A = torch.tensor(A, dtype=torch.float32,
                         requires_grad=False, device=device)
        A_hat = A + torch.eye(A.shape[0], A.shape[1], device=device)

        # Set the encoder to extract the hidden features.
        self.encoder = nn.Linear(n_in_features, n_hidden_features, bias=False)

        # Set the list of S-GNN modules.
        self.s_gnns = nn.ModuleList(
            [S_GNN(n_hidden_features, A_hat) for _ in range(n_in_timesteps)])

        # Set the list of hidden S-GNN modules.
        self.hidden_s_gnns = nn.ModuleList(
            [S_GNN(n_hidden_features, A_hat)
             for _ in range(n_in_timesteps - 1)])

        # Set the list of GRU modules.
        self.grus = nn.ModuleList(
            [GRU(n_hidden_features, n_hidden_features)
             for _ in range(n_in_timesteps)])

        # Set the positional encoder module.
        self.positional_encoder = PositionalEncoder(n_hidden_features,
                                                    n_in_timesteps)

        # Set the multi head attention module.
        self.transformer = Transformer(n_hidden_features, n_in_timesteps,
                                       n_attention_heads)

        # Set the convolutional layer to augment the number of
        # time steps channels to the output time steps.
        self.timesteps_convolution = nn.Conv2d(
            n_in_timesteps, n_out_timesteps, kernel_size=1, 
            bias=False) if n_in_timesteps != n_out_timesteps else None

        # Set the multi-layer prediction head.
        self.prediction_head = nn.Sequential(
            nn.Linear(n_hidden_features, n_hidden_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_hidden_features, n_out_features, bias=False)
        )

        # Push the model to the selected device.
        self.to(device)

    def forward(self, x: torch.FloatTensor):
        # Get the input dimensions.
        batch_size, len_timeseries, n_nodes, _ = x.shape

        # Set the base hidden state for the first GRU module.
        base_hidden_state = torch.zeros(
            [batch_size, n_nodes, self.n_hidden_features],
            dtype=torch.float32, device=x.device)

        # Encode the input by extracting hidden features.
        out = self.encoder(x)

        # Set the output list of the GRU modules results.
        outs = []

        for i in range(len_timeseries):
            # Get the S_GNN output state at the given timestamp.
            out_state = self.s_gnns[i](out[:,i])
            # Get the hidden state at the previous timestamp.
            if i > 0:
                hidden_state = self.hidden_s_gnns[i - 1](hidden_state)
            else:
                hidden_state = base_hidden_state
            # Get the GRU hidden state output at the given timestamp.
            hidden_state = self.grus[i](out_state, hidden_state)
            outs.append(hidden_state)

        # Stack the GRU modules results for each timestamp.
        out = torch.stack(outs, 1)

        # Get positional encoding and combine it to the result.
        out = self.positional_encoder(out)

        # Get the transformer layer results.
        out = self.transformer(out)

        # Apply the convolution to extract the output timesteps.
        if self.timesteps_convolution is not None:
            out = self.timesteps_convolution(out)

        # Predict the results.
        out = self.prediction_head(out)
        return out
