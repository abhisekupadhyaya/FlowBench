import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn as nn
from neuralop.models import TFNO
import glob

class LidDrivenDataset(Dataset):
    """
    Custom dataset for loading and processing Lid Driven Cavity problem data from .npz files.
    """
    def __init__(self, file_path_x, file_path_y, transform=None):
        """
        Initializes the dataset with the paths to the .npz files and an optional transform.
        
        Args:
            file_path_x (str): Path to the .npz file containing the input data.
            file_path_y (str): Path to the .npz file containing the target data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load data from .npz files
        x = np.load(file_path_x)['data']
        y = np.load(file_path_y)['data']
        
        # Convert numpy arrays to PyTorch tensors
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the sample and its label at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (sample, target) where sample is the input data and target is the expected output.
        """
        sample = self.x[idx]
        target = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target
    
    def create_dataloader(self, batch_size, split_fraction=0.8, shuffle=True):
        """
        Creates and returns data loaders for training and validation sets.

        Args:
            batch_size (int): Batch size for the data loaders.
            split_fraction (float, optional): Fraction of the dataset to use for training. Default is 0.8.
            shuffle (bool, optional): Whether to shuffle the dataset before splitting. Default is True.

        Returns:
            tuple: (train_loader, val_loader) where train_loader is the data loader for the training set
                   and val_loader is the data loader for the validation set.
        """
        dataset_size = len(self)
        train_size = int(dataset_size * split_fraction)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(self, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader, val_loader

class TensorizedFNO(nn.Module):
    """
    Tensorized Fourier Neural Operator (TFNO) model for learning mappings between function spaces.
    """
    def __init__(self, n_modes, hidden_channels, in_channels, out_channels, projection_channels):
        """
        Initializes the TFNO model with specified parameters.

        Args:
            n_modes (tuple): Number of modes for Fourier layers.
            hidden_channels (int): Number of hidden channels.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(TensorizedFNO, self).__init__()
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projection_channels = projection_channels

        # Initialize the TFNO with Tucker factorization
        self.tfno = TFNO(n_modes=self.n_modes, hidden_channels=self.hidden_channels,
                         in_channels=self.in_channels, out_channels=self.out_channels, 
                         projection_channels = self.projection_channels)

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
        Saves the model weights to a checkpoint file.

        Args:
            save_name (str): Name of the checkpoint file.
            save_folder (str, optional): Folder to save the checkpoint. Defaults to 'experiments/fno'.
        """
        os.makedirs(save_folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_folder, f'{save_name}.pth'))

    def load_checkpoint(self, save_name=None, save_folder='../../experiments/fno/checkpoints'):
        """
        Loads the model weights from a checkpoint file.

        Args:
            save_name (str, optional): Name of the checkpoint file. If None, loads the latest checkpoint.
            save_folder (str, optional): Folder containing the checkpoint. Defaults to 'experiments/fno'.
        """
        if save_name is None:
            # Load the latest checkpoint based on the modification time
            checkpoints = glob.glob(os.path.join(save_folder, '*.pth'))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found in the specified folder.")
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        else:
            latest_checkpoint = os.path.join(save_folder, f'{save_name}.pth')

        self.load_state_dict(torch.load(latest_checkpoint))

# Example usage:
# model = TensorizedFNO(n_modes=(16, 16), hidden_channels=64, in_channels=1, out_channels=1)
# model.save_checkpoint(save_name='model_checkpoint')
# model.load_checkpoint()


if torch.cuda.is_available():
    # Set the default device to CUDA
    device = torch.device('cuda')
    #torch.set_default_device(device)
    print('Using CUDA for tensor operations')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU for tensor operations')
    

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, epochs, device, log_file=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.log_file = log_file

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model.forward(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        best_val_loss = float('inf')

        #if self.log_file:
        #    log_dir = os.path.dirname(self.log_file)
        #    os.makedirs(log_dir, exist_ok=True)
        #    log_file_handle = open(self.log_file, 'w')

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            log_line = f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}'
            print(log_line)

            #if self.log_file_handle:
            #    log_file_handle.write(log_line + '\n')

            #if (epoch+1) % 5 == 0:
            #    best_val_loss = val_loss
            #    self.model.save_checkpoint(save_name=str(epoch+1))

        #if log_file_handle:
        #    log_file_handle.close()

        print("Training complete.")

    def load_model(self):
        self.model.load_checkpoint()

class TrainFNO(Trainer):
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, epochs, device, loss_save = '../../experiments/fno/loss.txt'):
        super().__init__(model, optimizer, loss_fn, train_loader, val_loader, epochs, device, loss_save)
        if not isinstance(model, TensorizedFNO):
            raise TypeError("The model should be an instance of TensorizedFNO")
            
LidDriven_dataset = LidDrivenDataset(
   file_path_x='../../../rtali/projects/all-neural-operators/TimeDependentNS/LidData_Curated_Input/harmonics/harmonics_lid_driven_cavity_X.npz',
   file_path_y='../../../rtali/projects/all-neural-operators/TimeDependentNS/LidData_Curated_Input/harmonics/harmonics_lid_driven_cavity_y.npz'
)


print(2)

# Create data loaders for training and validation
train_loader, val_loader = LidDriven_dataset.create_dataloader(batch_size=10, split_fraction=0.8, shuffle=True)


print(3)

# Create an instance of the TensorizedFNO model
model = TensorizedFNO(n_modes=(32, 32), in_channels=3, out_channels=3, hidden_channels=64, projection_channels=128).to(device)

# Set the learning rate and number of epochs
learning_rate = 0.001
num_epochs = 2000

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print(4)

# Create an instance of the TrainFNO class
#FNO_trainer = TrainFNO(model=model, optimizer=optimizer, loss_fn=criterion,
#                      train_loader=train_loader, val_loader=val_loader, epochs=num_epochs,
#                      device=device)

# Training loop
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0

    # Iterate over batches
    for batch_data, batch_targets in train_loader:
        # Move data to device
        batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_data)

        # Compute the loss
        loss = criterion(outputs, batch_targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average training loss for the epoch
    train_loss = running_loss / len(train_loader)

    if (epoch + 1) % 10 == 0:
        # Print the loss for each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {train_loss:.8e} ')
        #checkpoint_path = f"FNO_checkpoint_epoch_{epoch+1}.pt"
        #torch.save(model.state_dict(), checkpoint_path)

print('Finished Training')