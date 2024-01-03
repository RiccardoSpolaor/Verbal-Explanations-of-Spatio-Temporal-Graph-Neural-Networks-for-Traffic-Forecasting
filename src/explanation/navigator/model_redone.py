from typing import Union
import torch
from torch import nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.typing import Adj, OptPairTensor, Size
import numpy as np


from ...spatial_temporal_gnn.modules import GRU
from torch_geometric_temporal.nn.attention.mtgnn import MTGNN


class Navigator(nn.Module):
    """
    A navigator model used to predict the correlation score of an encoded
    input event with respect to a single encoded target event.
    It takes the concatenation of the input event and the target event
    and computes the correlation score between them.

    Attributes
    ----------
    linear_encoder : LazyLinear
        Linear encoder to extract the hidden features of the
        concatenated input and target events.
    linear_decoder : Linear
        Linear decoder to decode the hidden features to the logit
        prediction.
    device : str
        The device that is used for training and querying the model.
        
    Methods
    -------
    forward(input_event: FloatTensor, target_event: FloatTensor
    ) -> FloatTensor
        Computes the forward pass of the model.
    """
    def __init__(self, device: str, A: np.ndarray, hidden_features: int = 64) -> None:
        """Initialize the navigator model.

        Parameters
        ----------
        device : str
            The device that is used for training and querying the model.
        hidden_features : int, optional
            The number of hidden features, by default 64.
        """
        super().__init__()

        # Get the refined adjacency matrix.
        A = torch.tensor(
            A,
            dtype=torch.float32,
            device=device)
        A_hat = A + torch.eye(A.shape[0], A.shape[1], device=device)
        
        self.AA = A
        self.A = A.nonzero().t().contiguous().long()
        self.W = torch.stack([A[self.A[0, i], self.A[1, i]] for i in range(self.A.shape[1])])
        #self.mtgnn = MTGNN()

        # Set the list of S-GNN modules.
        #self.s_gnns = nn.ModuleList(
        #    [S_GNN(9, A_hat) for _ in range(12)])
        
        self.sages = nn.ModuleList([GraphSAGE(9, 64, 9) for _ in range(12)])
        self.grus = nn.ModuleList(
            [GRU(9, 9)
             for _ in range(12)])
        #self.grus = nn.ModuleList[GRU(9, 9) for _ in range(12)]

        # Set the linear encoder in combination with the S-GNN modules.
        self.linear_encoder = nn.LazyLinear(hidden_features)
        # Set the linear decoder.
        self.linear_decoder = nn.Linear(hidden_features, 1)
        # Set the device that is used for training and querying the model.
        self.device = device
        self.to(device)
        
        self.hidden_features = hidden_features

    def forward(
        self,
        input_event: torch.FloatTensor,
        graph_at_instance: torch.FloatTensor,
        target_event: torch.FloatTensor) -> torch.FloatTensor:
        """The forward pass of the navigator model.

        Parameters
        ----------
        input_event : FloatTensor
            The input event.
        target_event : FloatTensor
            The target event.

        Returns
        -------
        FloatTensor
            The logit prediction of the correlation score between the
            input event and the target event.
        """
        '''xs = [self.s_gnns[i](g) for i, g in enumerate(graph_at_instance)]
        z = []
        for i in input_event:
            z.append(xs[int(i[0].item())][int(i[1].item())])
        z = torch.stack(z)    
        # Concatenate the input event and the target event.
        x = torch.cat((z, target_event), dim=1)'''
        #ts = input_event[:, i]à
        
        # Get the input dimensions.
        b, t, n_nodes, _ = graph_at_instance.shape

        # Set the base hidden state for the first GRU module.
        base_hidden_state = torch.zeros(
            [b, n_nodes, 9],
            dtype=torch.float32,
            device=self.device)
        
        out = graph_at_instance
        outs = []
        #sages = [s(graph_at_instance[i], self.A, self.W) for i, s in enumerate(self.sages)]
        for i in range(t):
            # Get the S_GNN output state at the given timestamp.
            out_state = self.sages[i](out[:, i], self.A, self.W)
            # Get the hidden state at the previous timestamp.
            if i == 0:
                hidden_state = base_hidden_state
            # Get the GRU hidden state output at the given timestamp.
            hidden_state = self.grus[i](out_state, hidden_state)
            outs.append(hidden_state)

        outs = torch.stack(outs, dim=1)
        #print(outs.shape)

        #xs = []
        
        #for e in input_event:
        #    xs.append(outs[e[0]][e[1]])
        
        #xs = torch.stack(xs)
        # Repeat target event as the size of outs
        # from shape (3, 11) bring to (3, 12, 207, 11)
        # Create a matrix of size (b, t, n_nodes, f) with the target event of size (b, f) repeated 
        
        target_event = target_event.unsqueeze(1).unsqueeze(1).repeat(1, t, n_nodes, 1)
        #print(outs.shape, target_event.shape)
        x = torch.cat((outs, target_event), dim=-1)
        # Encode the concatenated events.
        out = self.linear_encoder(x)
        # Decode the output to get the logit prediction.
        out = self.linear_decoder(out)
        #out = torch.sigmoid(out)
        out = out * (graph_at_instance[..., 0:1] != 0).float()
        # Apply sigmoid for each element in the batch separately.
        return out #torch.sigmoid(out)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConvWithWeights(in_dim, hidden_dim)
        self.conv2 = SAGEConvWithWeights(hidden_dim, hidden_dim)
        self.conv3 = SAGEConvWithWeights(hidden_dim, out_dim)
    
    def forward(self, x, adj, weights):
        x = self.conv1(x, adj, weights)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, adj, weights)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv3(x, adj, weights)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        return x
    
class SAGEConvWithWeights(SAGEConv):
    def forward(
        self,
        x: Union[torch.Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: torch.Tensor,
        size: Size = None
        ) -> torch.Tensor:

        if isinstance(x, torch.Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, edge_weight=edge_weight, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    
'''class SAGEConv(torch.nn.Module):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    If :obj:`project = True`, then :math:`\mathbf{x}_j` will first get
    projected via

    .. math::
        \mathbf{x}_j \leftarrow \sigma ( \mathbf{W}_3 \mathbf{x}_j +
        \mathbf{b})

    as described in Eq. (3) of the paper.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (string or Aggregation, optional): The aggregation scheme to use.
            Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
            *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`.
            (default: :obj:`"mean"`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        project (bool, optional): If set to :obj:`True`, the layer will apply a
            linear transformation followed by an activation function before
            aggregation (as described in Eq. (3) of the paper).
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = nn-LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = nn.Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.aggr_module.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, torch.Tensor):
            x: Tuple[Tensor, Optional[Tensor]] = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out'''