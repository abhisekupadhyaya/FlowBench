import os
import torch
import torch.nn as nn
import glob
from models.Modules.CNO.CNOModule import CNO
from neuralop.models import TFNO

class NeuralModel(nn.Module):
    """
    Base model class providing common functionality for saving and loading checkpoints.
    """
    def __init__(self):
        super(NeuralModel, self).__init__()

    def save_checkpoint(self, save_name, save_folder):
        """
        Saves the model weights to a checkpoint file.

        Args:
            save_name (str): Name of the checkpoint file.
            save_folder (str): Folder to save the checkpoint.
        """
        os.makedirs(save_folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_folder, f'{save_name}.pth'))

    def load_checkpoint(self, save_name=None, save_folder=None):
        """
        Loads the model weights from a checkpoint file.

        Args:
            save_name (str, optional): Name of the checkpoint file. If None, loads the latest checkpoint.
            save_folder (str, optional): Folder containing the checkpoint. Defaults to provided save_folder.
        """
        if save_folder is None:
            raise ValueError("save_folder must be specified.")

        if save_name is None:
            # Load the latest checkpoint based on the modification time
            checkpoints = glob.glob(os.path.join(save_folder, '*.pth'))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found in the specified folder.")
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        else:
            latest_checkpoint = os.path.join(save_folder, f'{save_name}.pth')

        self.load_state_dict(torch.load(latest_checkpoint))

class CompressedCNO(NeuralModel):
    def __init__(self, in_dim, out_dim, N_layers, in_size, out_size):
        super(CompressedCNO, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.N_layers = N_layers
        self.in_size = in_size
        self.out_size = out_size

        # Initialize the CNO with Tucker factorization
        self.cno = CNO(in_dim=self.in_dim, 
                       out_dim=self.out_dim,
                       N_layers=self.N_layers,
                       in_size=self.in_size,
                       out_size=self.out_size)

    def forward(self, x):
        """
        Forward pass of the CNO model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the CNO.
        """
        return self.cno(x)

    def save_checkpoint(self, save_name, save_folder='../../experiments/cno/checkpoints'):
        """
        Saves the model weights to a checkpoint file specific to CNO.

        Args:
            save_name (str): Name of the checkpoint file.
            save_folder (str, optional): Folder to save the checkpoint. Defaults to '../../experiments/cno/checkpoints'.
        """
        super().save_checkpoint(save_name, save_folder)

    def load_checkpoint(self, save_name=None, save_folder='../../experiments/cno/checkpoints'):
        """
        Loads the model weights from a checkpoint file specific to CNO.

        Args:
            save_name (str, optional): Name of the checkpoint file. If None, loads the latest checkpoint.
            save_folder (str, optional): Folder containing the checkpoint. Defaults to '../../experiments/cno/checkpoints'.
        """
        super().load_checkpoint(save_name, save_folder)

class TensorizedFNO(NeuralModel):
    def __init__(self, n_modes, hidden_channels, in_channels, out_channels, projection_channels):
        super(TensorizedFNO, self).__init__()
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projection_channels = projection_channels

        # Initialize the TFNO with Tucker factorization
        self.tfno = TFNO(n_modes=self.n_modes, hidden_channels=self.hidden_channels,
                         in_channels=self.in_channels, out_channels=self.out_channels, 
                         projection_channels=self.projection_channels)

    def forward(self, x):
        """
        Forward pass of the TFNO model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the TFNO.
        """
        return self.tfno(x)

    def save_checkpoint(self, save_name, save_folder='../../experiments/fno/checkpoints'):
        """
        Saves the model weights to a checkpoint file specific to FNO.

        Args:
            save_name (str): Name of the checkpoint file.
            save_folder (str, optional): Folder to save the checkpoint. Defaults to '../../experiments/fno/checkpoints'.
        """
        super().save_checkpoint(save_name, save_folder)

    def load_checkpoint(self, save_name=None, save_folder='../../experiments/fno/checkpoints'):
        """
        Loads the model weights from a checkpoint file specific to FNO.

        Args:
            save_name (str, optional): Name of the checkpoint file. If None, loads the latest checkpoint.
            save_folder (str, optional): Folder containing the checkpoint. Defaults to '../../experiments/fno/checkpoints'.
        """
        super().load_checkpoint(save_name, save_folder)

# Example usage:
# model_cno = CompressedCNO(in_dim=11, out_dim=10, N_layers=5, in_size=64, out_size=64)
# model_cno.save_checkpoint(save_name='model_checkpoint')
# model_cno.load_checkpoint()

# model_tfno = TensorizedFNO(n_modes=(16, 16), hidden_channels=64, in_channels=1, out_channels=1, projection_channels=32)
# model_tfno.save_checkpoint(save_name='model_checkpoint')
# model_tfno.load_checkpoint()
