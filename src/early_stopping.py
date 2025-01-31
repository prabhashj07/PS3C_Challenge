import os
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False, save_dir='artifacts', model_name='best_model.pth'):
        """
        Early stopping to stop training if validation loss doesn't improve.
        
        Args:
            patience (int): Number of epochs to wait for improvement.
            delta (float): Minimum change to qualify as an improvement.
            verbose (bool): Whether to print messages when stopping early.
            save_dir (str): Directory to save the best model checkpoint.
            model_name (str): Name of the saved model file.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.val_loss_min = np.inf
        
        # Create the save directory if it doesn't exist
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Define the path for saving the model
        self.path = os.path.join(self.save_dir, model_name)

    def __call__(self, val_loss, model):
        """
        Check if the validation loss has improved and save the model if needed.
        
        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model to save if validation loss improves.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased: {self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model...')
            torch.save(model.state_dict(), self.path)  # Save the model
        else:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss did not improve: {val_loss:.6f}. Patience counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")
