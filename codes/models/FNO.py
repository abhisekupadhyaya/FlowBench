import torch
import torch.nn as nn
from neuralop.models import TFNO

class TensorizedFNO(nn.Module):
    """
    Tensorized Fourier Neural Operator (TFNO) model for learning mappings between function spaces.
    This implementation uses Tucker factorization for efficient computation.
    """
    def __init__(self, n_modes, hidden_channels, in_channels, out_channels, rank_factor):
        """
        Initializes the TFNO model with specified parameters.

        Args:
            n_modes (tuple): Number of modes for Fourier layers.
            hidden_channels (int): Number of hidden channels.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            rank_factor (float): Factor for Tucker factorization, representing the rank as a fraction of the full rank.
        """
        super(TensorizedFNO, self).__init__()
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank_factor = rank_factor

        # Initialize the TFNO with Tucker factorization
        self.tfno = TFNO(n_modes=self.n_modes, hidden_channels=self.hidden_channels,
                         in_channels=self.in_channels, out_channels=self.out_channels,
                         factorization='tucker', implementation='factorized', rank=self.rank_factor)

    def forward(self, x):
        """
        Forward pass of the TFNO model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the TFNO.
        """
        return self.tfno(x)

    def save_checkpoint(self, save_folder, save_name):
        """
        Saves the model weights to a checkpoint file.

        Args:
            save_folder (str): Folder to save the checkpoint.
            save_name (str): Name of the checkpoint file.
        """
        torch.save(self.state_dict(), f'{save_folder}/{save_name}.pth')

    def load_checkpoint(self, save_folder, save_name):
        """
        Loads the model weights from a checkpoint file.

        Args:
            save_folder (str): Folder containing the checkpoint.
            save_name (str): Name of the checkpoint file.
        """
        self.load_state_dict(torch.load(f'{save_folder}/{save_name}.pth'))
