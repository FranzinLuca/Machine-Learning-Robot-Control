import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import DynamicMLP
from src.train import train_one_epoch, validate
from src.config import Config
from torch.utils.tensorboard import SummaryWriter

def get_optimizer(trial, model):
    """Selects optimizer based on trial suggestion."""
    lr = trial.suggest_float("lr", Config.LR_RANGE[0], Config.LR_RANGE[1], log=True)
    return optim.Adam(model.parameters(), lr=lr)

def objective(trial, train_loader, val_loader, input_dim, output_dim, device):
    """Optuna objective function for hyperparameter optimization."""
    n_layers = trial.suggest_int("n_layers", Config.NUM_LAYERS_RANGE[0], Config.NUM_LAYERS_RANGE[1])
    dropout = trial.suggest_float("dropout", Config.DROPOUT_RANGE[0], Config.DROPOUT_RANGE[1])
    activation = trial.suggest_categorical("activation", Config.ACTIVATION_OPTIONS)
    use_residual = trial.suggest_categorical("use_residual", [True, False])
    hidden_layers = [trial.suggest_categorical(f"hidden_units_layer_{i}", Config.HIDDEN_UNITS_OPTIONS) for i in range(n_layers)]

    model = DynamicMLP(input_dim, output_dim, hidden_layers, dropout, activation, use_residual).to(device)
    optimizer = get_optimizer(trial, model)
    criterion = nn.MSELoss()
    
    epochs = Config.EPOCHS_OPTUNA 
    
    run_name = f"runs/trial_{trial.number}"
    writer = SummaryWriter(run_name)
    
    val_loss = float('inf')
    
    try:
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
            val_loss, val_r2 = validate(model, val_loader, criterion, device, epoch, writer)
            trial.report(val_loss, epoch)
            writer.add_scalar('Optuna/R2_Score', val_r2, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    finally:
        writer.close()
            
    return val_loss