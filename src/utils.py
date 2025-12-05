import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import json

def set_seed(seed=42):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")

def compute_metrics(y_true, y_pred, raw_scale=False):
    """
    Computes R2 and MSE.
    Args:
        y_true: Ground truth (numpy array)
        y_pred: Predictions (numpy array)
        raw_scale: If True, indicates inputs are unscaled (for reporting).
    Returns:
        dict: {'r2': float, 'mse': float}
    """
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {"r2": r2, "mse": mse}

def plot_loss_curves(train_losses, val_losses, title="Training Progress", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_results_publication(y_true, y_pred, joint_idx=0, robot_name="Reacher3", save_dir="plots"):
    """
    Generates publication-quality Parity and Residual plots.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create a figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Parity Plot ---
    # Lower alpha to see density
    ax1.scatter(y_true[:, joint_idx], y_pred[:, joint_idx], alpha=0.15, s=10, color='tab:blue')
    
    # Perfect prediction line
    min_val = min(y_true[:, joint_idx].min(), y_pred[:, joint_idx].min())
    max_val = max(y_true[:, joint_idx].max(), y_pred[:, joint_idx].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    
    ax1.set_xlabel(f'True dq_{joint_idx} (rad)', fontsize=12)
    ax1.set_ylabel(f'Predicted dq_{joint_idx} (rad)', fontsize=12)
    ax1.set_title(f'{robot_name}: Parity Plot (Joint {joint_idx})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Residual Plot ---
    # Residuals = True - Predicted
    residuals = y_true[:, joint_idx] - y_pred[:, joint_idx]
    
    ax2.scatter(y_pred[:, joint_idx], residuals, alpha=0.15, s=10, color='tab:green')
    ax2.axhline(0, color='red', linestyle='--', lw=2)
    
    ax2.set_xlabel(f'Predicted dq_{joint_idx} (rad)', fontsize=12)
    ax2.set_ylabel('Residual (Error)', fontsize=12)
    ax2.set_title(f'{robot_name}: Residuals (Joint {joint_idx})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to your plots folder
    save_path = os.path.join(save_dir, f"{robot_name}_joint{joint_idx}_analysis.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved analysis plot to {save_path}")
    plt.close()
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, min_delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)

def save_artifacts(model, x_scaler, y_scaler, save_dir="models", model_name="best_model"):
    """
    Saves model weights, architecture config, and scalers.
    
    Args:
        model: The trained PyTorch model.
        hyperparams (dict): Dictionary containing 'input_dim', 'output_dim', 'hidden_layers', 'activation', etc.
        x_scaler, y_scaler: Fitted sklearn scalers.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Weights
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
    
    # Save Scalers
    joblib.dump(x_scaler, os.path.join(save_dir, f"{model_name}_x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(save_dir, f"{model_name}_y_scaler.pkl"))
    
    print(f"Artifacts saved to {save_dir}/{model_name}*")