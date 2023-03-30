"""
Module containing the modules of the Spatial-Temporal Graph
Neural Network pytorch model.
"""
import math
import torch
from torch import nn


class S_GNN(nn.Module):
    """
    A Spatial GNN module to capture the spatial relations of an
    instance of a graph of a sequence.

    Attributes
    ----------
    latent_encoder : Sequential
        Module to obtain the latent representation of the input.
    linear : Linear
        Linear layer to model the spatial feature extraction.
    A_hat : FloatTensor
        Refined adjacency matrix.
    """
    def __init__(self, n_features: int, A_hat: torch.FloatTensor) -> None:
        """
        Initialize the Spatial GNN module.

        Arguments
        ---------
        n_features : int
            Number of input features.
        A_hat : FloatTensor
            Refined adjacency matrix.
        """
        super().__init__()
        # Module to obtain the latent representation of the input.
        self.latent_encoder = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.Linear(n_features, n_features // 2, bias=False)
        )

        # Linear layer to model the spatial feature extraction.
        self.linear = nn.Linear(n_features, n_features, bias=False)

        # Set the refined adjacency matrix.
        self.A_hat = A_hat

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply the forward pass to the Spatial GNN module to the input.

        Arguments
        ---------
        x : FloatTensor
            The input tensor.

        Returns
        -------
        FloatTensor
            The output tensor.
        """
        # Get the latent representation of the input.
        p = self.latent_encoder(x)

        # Apply the score function.
        S = p @ p.transpose(-1, -2)
        # Stabilize score function to avoid softmax overflowing.
        S_stabilized = S - torch.max(S, dim=-1, keepdim=True).values

        # Get the pair-wise relation between any road node.
        R = torch.softmax(S_stabilized, -1)
        #R = torch.softmax(S, -1)

        # Get the sparsified relation matrix.
        A_hat = self.A_hat.expand_as(R)
        R_hat = R * (A_hat > 0).float() 
        R_hat += torch.eye(R_hat.shape[-2], R_hat.shape[-1], device=x.device)

        # Get refined degree matrix from the sparsified relation matrix.
        D_hat = R_hat.sum(-1) ** -.5
        # D_hat[torch.isinf(D_hat)] = 0.
        D_hat = torch.diag_embed(D_hat)

        # Apply the modified GCN operation.
        A = D_hat @ R_hat @ D_hat
        return torch.relu(self.linear(A @ x))

class GRU(nn.Module):
    """
    Apply a Gated Recurrent Unit (GRU) RNN to a hidden representation
    of an instance of a graph of a sequence.
    
    Attributes
    ----------
    z_x_linear : Linear
        Update gate layer for the input feature.
    z_h_linear : Linear
        Update gate layer for the hidden feature.
    r_x_linear : Linear
        Reset gate layer for the input feature
    r_h_linear : Linear
        Reset gate layer for the hidden feature
    h_x_linear : Linear
        State gate layer for the input feature
    h_h_linear : Linear 
        State gate layer for the hidden feature
    """
    def __init__(self, n_input_features: int, n_hidden_features: int) -> None:
        """Initialize the GRU module.

        Parameters
        ----------
        n_input_features : int
            Number of features of the input state.
        n_hidden_features : int
            Number of features of the hidden state.
        """
        super().__init__()
        # Update gate layers.
        self.z_x_linear = nn.Linear(n_input_features, n_hidden_features,
                                    bias=False)
        self.z_h_linear = nn.Linear(n_hidden_features, n_hidden_features,
                                    bias=False)

        # Reset gate layers.
        self.r_x_linear = nn.Linear(n_input_features, n_hidden_features,
                                    bias=False)
        self.r_h_linear = nn.Linear(n_hidden_features, n_hidden_features,
                                    bias=False)

        # State gate layers.
        self.h_x_linear = nn.Linear(n_input_features, n_hidden_features,
                                    bias=False)
        self.h_h_linear = nn.Linear(n_hidden_features, n_hidden_features,
                                    bias=False)

    def forward(self, x: torch.FloatTensor, h: torch.FloatTensor
                ) -> torch.FloatTensor:
        """
        Compute the forward pass of the GRU module.
        
        Arguments
        ---------
        x : FloatTensor
            Input tensor.
        h : FloatTensor
            Hidden state tensor.
        
        Returns
        -------
        FloatTensor
            Output tensor.
        """
        # Update Gate.
        z_x = self.z_x_linear(x)
        z_h = self.z_h_linear(h)
        z_t = torch.sigmoid(z_x + z_h)

        # Reset Gate.
        r_x = self.r_x_linear(x)
        r_h = self.r_h_linear(h)
        r_t = torch.sigmoid(r_x + r_h)

        # State gate.
        h_x = self.h_x_linear(x)
        h_h = self.h_h_linear(h)
        h_t = torch.tanh(h_x + r_t * h_h)

        # Get GRU output.
        return (1 - z_t) * h_t + z_t * h

class Transformer(nn.Module):
    """
    Apply the multi-head attention mechanism to the hidden representations
    of the graph sequences for a global understanding of the time
    relation.

    Attributes
    ----------
    n_attention_heads: int
        Number of attention heads.
    queries_linear: Linear
        Linear layer to model the queries.
    keys_linear: Linear
        Linear layer to model the keys.
    values_linear: Linear
        Linear layer to model the values.
    normalization: BatchNorm2d
        Normalization layer.
    normalization_out: BatchNorm2d
        Output normalization layer.
    feed_forward: Sequential
        Multi-layer feed forward module.
    """
    def __init__(self, n_features: int, n_timesteps: int,
                 n_attention_heads: int) -> None:
        """Initialize the Transformer layer.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_timesteps : int
            Number of time steps.
        n_attention_heads : int
            Number of attention heads.
        """
        super().__init__()
        # Set the number of attention heads.
        self.n_attention_heads = n_attention_heads

        # Linear layers to model the queries, keys and values.
        self.queries_linear = nn.Linear(n_features, n_features, bias=False)
        self.keys_linear = nn.Linear(n_features, n_features, bias=False)
        self.values_linear = nn.Linear(n_features, n_features, bias=False)

        # Normalization layers.
        self.normalization = nn.BatchNorm2d(
            n_timesteps, track_running_stats=False)
        self.normalization_out = nn.BatchNorm2d(
            n_timesteps, track_running_stats=False)

        # Multi-layer feed forward module.
        self.feed_forward = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, n_features, bias=False)
        )

    def forward(self, x):
        """
        Compute the forward pass of the Transformer layer.
        
        Parameters
        ----------
        x : FloatTensor
            The input tensor.
        
        Returns
        -------
        FloatTensor
            The output tensor.
        """
        # Get queries, keys and values.
        Q = self.queries_linear(x)
        K = self.keys_linear(x)
        V = self.values_linear(x)

        # Split Q, K and V features according to the attention head number.
        # Concatenate the attention heads row-wise.
        Q_h = torch.cat(torch.split(Q, self.n_attention_heads, -1), 0)
        K_h = torch.cat(torch.split(K, self.n_attention_heads, -1), 0)
        V_h = torch.cat(torch.split(V, self.n_attention_heads, -1), 0)

        # Reshape the matrices in order to apply the operations node-wise.
        permutation = [0, 2, 1, 3]
        Q_h = Q_h.permute(*permutation)
        K_h = K_h.permute(*permutation)
        V_h = V_h.permute(*permutation)

        # Apply the multi-head attention mechanism.
        H = Q_h @ K_h.transpose(-2, -1) / (self.n_attention_heads ** .5)
        H = torch.softmax(H, -1)
        H = H @ V_h

        # Split the result according to the batch size and re-concatenate.
        out = torch.cat(torch.split(H, x.shape[0], 0), -1)
        # Reshape the matrix to the original form.
        out = out.permute(*permutation)

        # Apply residual connection and batch normalization.
        out += x
        norm = self.normalization(out)

        # Apply feed forward module.
        out = self.feed_forward(norm)

        # Apply residual connection and batch normalization.
        out += norm
        out = self.normalization_out(out)

        return out

class PositionalEncoder(nn.Module):
    """
    Positional encoding module for time series data.
    
    Attributes
    ----------
    pe : FloatTensor
        Positional encodings tensor.
        
    """
    def __init__(self, n_features: int, n_timesteps: int) -> None:
        """Initialize the positional encoding module.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_timesteps : int
            Number of timesteps.
        """
        super().__init__()
        # Initialize the positional encoder.
        positional_encoder = torch.zeros(n_timesteps, n_features)

        # Get the positions with respect to the timeseries.
        positions = torch.arange(n_timesteps).unsqueeze(1)

        # Get the divisor term for the positional encoding.
        divisor = torch.exp(torch.arange(0, n_features, 2) *
                            (math.log(10_000.) / n_features))

        # Compute the positional encodings.
        positional_encoder[:, 0::2] = torch.sin(positions * divisor)
        positional_encoder[:, 1::2] = torch.cos(positions * divisor)

        # Reshape to consider batch and feature dimensions.
        positional_encoder = positional_encoder.unsqueeze(0).unsqueeze(2)

        # Register as a non-parameter.
        self.register_buffer('pe', positional_encoder)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the forward pass of the Positional Encoding module.

        Parameters
        ----------
        x : FloatTensor
            The input tensor.

        Returns
        -------
        FloatTensor
            The output tensor.
        """
        return x + torch.autograd.Variable(self.pe, requires_grad=False)
