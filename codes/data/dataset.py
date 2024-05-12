# Custom dataset definitions and data loaders
from torch.utils.data import Dataset
import torch
from scipy.io import loadmat
import numpy as np

class CustomDataset(Dataset):
    """
    Custom dataset for loading and processing data from a .mat file.
    """
    def __init__(self, file_path, transform=None):
        """
        Initializes the dataset with the path to the .mat file and an optional transform.
        
        Args:
            file_path (str): Path to the .mat file containing the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load data from .mat file
        data_mat = loadmat(file_path)
        
        # Assuming 'u' and 'a' are keys in the .mat file that hold the data
        u_numpy = data_mat['u']
        a_numpy = data_mat['a']
        
        # Convert numpy arrays to PyTorch tensors
        self.u_tensor = torch.tensor(u_numpy, dtype=torch.float32)
        self.a_tensor = torch.tensor(a_numpy, dtype=torch.float32)
        
        # Concatenate along the desired dimension and permute to match input dimensions for the model
        self.concatenated_tensor = torch.cat((self.u_tensor[:,:,:,:10], self.a_tensor.unsqueeze(-1)), dim=-1).permute(0, 3, 1, 2)
        self.result_tensor = self.u_tensor[:,:,:,:10].permute(0, 3, 1, 2)
        
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.concatenated_tensor)

    def __getitem__(self, idx):
        """
        Retrieves the sample and its label at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (sample, target) where sample is the input data and target is the expected output.
        """
        sample = self.concatenated_tensor[idx]
        target = self.result_tensor[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target