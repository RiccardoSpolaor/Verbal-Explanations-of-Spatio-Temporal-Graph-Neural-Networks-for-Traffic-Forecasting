import torch
from torch import nn

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
    def __init__(self, device: str, hidden_features: int = 64) -> None:
        """Initialize the navigator model.

        Parameters
        ----------
        device : str
            The device that is used for training and querying the model.
        hidden_features : int, optional
            The number of hidden features, by default 64.
        """
        super().__init__()
        # Set the linear encoder.
        self.linear_encoder = nn.LazyLinear(hidden_features)
        # Set the linear decoder.
        self.linear_decoder = nn.Linear(hidden_features, 1)
        # Set the device that is used for training and querying the model.
        self.device = device
        self.to(device)

    def forward(
        self, input_event: torch.FloatTensor,
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
        # Concatenate the input event and the target event.
        x = torch.cat((input_event, target_event), dim=1)
        # Encode the concatenated events.
        out = self.linear_encoder(x)
        # Decode the output to get the logit prediction.
        out = self.linear_decoder(out)
        return out
