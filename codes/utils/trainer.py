import os
import torch

from codes.models.FNO import TensorizedFNO
#from codes.models.models import CompressedCNO

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

        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            os.makedirs(log_dir, exist_ok=True)
            log_file_handle = open(self.log_file, 'w')

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            log_line = f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}'
            print(log_line)

            if self.log_file_handle:
                log_file_handle.write(log_line + '\n')

            if (epoch+1) % 5 == 0:
                self.model.save_checkpoint(save_name=str(epoch+1))

        if log_file_handle:
            log_file_handle.close()

        print("Training complete.")

    def load_model(self):
        self.model.load_checkpoint()

class TrainFNO(Trainer):
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, epochs, device, loss_save = '../../experiments/fno/loss.txt'):
        super().__init__(model, optimizer, loss_fn, train_loader, val_loader, epochs, device, loss_save)
        if not isinstance(model, TensorizedFNO):
            raise TypeError("The model should be an instance of TensorizedFNO")