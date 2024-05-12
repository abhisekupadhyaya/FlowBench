import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import CustomDataset
from models.model_tfno import TensorizedFNO
from utils.visualization import animate_tensor
import matplotlib.pyplot as plt

# Configuration parameters
file_path = 'path/to/your/dataset/NavierStokes_V1e-5_N1200_T20.mat'
batch_size = 64
learning_rate = 0.001
num_epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset = CustomDataset(file_path=file_path)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = TensorizedFNO(n_modes=(32, 32), hidden_channels=64, in_channels=11, out_channels=10, rank_factor=0.5).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_data, batch_targets in train_loader:
        batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
        
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print average loss per epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Optional: Visualize the first batch's output every few epochs
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            model.eval()
            sample_output = model(batch_data[:1]).detach()
            animate_tensor(sample_output)
            plt.show()

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'model_tfno.pth')