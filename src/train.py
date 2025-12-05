import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_index, writer=None):
    model.train()
    running_loss = 0.0
    
    desc = f"Epoch {epoch_index} [Train]"
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if writer is not None and i % 10 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            global_step = (epoch_index - 1) * len(loader) + i
            writer.add_scalar('Gradients/Global_Norm', total_norm, global_step)

        optimizer.step()
        
        current_loss = loss.item()
        running_loss += current_loss * inputs.size(0)
        pbar.set_postfix(loss=f"{current_loss:.4f}")
        
        if writer is not None:
            global_step = (epoch_index - 1) * len(loader) + i
            writer.add_scalar('Loss/Train_Batch', current_loss, global_step)

    epoch_loss = running_loss / len(loader.dataset)
    
    if writer is not None:
        writer.add_scalar('Loss/Train_Epoch', epoch_loss, epoch_index)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Hyperparameters/Learning_Rate', current_lr, epoch_index)
        
    return epoch_loss

def validate(model, loader, criterion, device, epoch_index, writer=None):
    """
    Evaluates the model on the validation set and logs result to TensorBoard.
    Returns: val_loss, val_r2
    """
    model.eval()
    running_loss = 0.0
    
    # Store all predictions/targets for global R2 calculation
    all_targets = []
    all_outputs = []
    
    desc = f"Epoch {epoch_index} [Val]"
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            all_targets.append(targets.cpu())
            all_outputs.append(outputs.cpu())
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    epoch_loss = running_loss / len(loader.dataset)
    
    # --- R2 Score Calculation ---
    all_targets = torch.cat(all_targets).numpy()
    all_outputs = torch.cat(all_outputs).numpy()
    val_r2 = r2_score(all_targets, all_outputs, multioutput='uniform_average')

    # --- TENSORBOARD ---
    if writer is not None:
        writer.add_scalar('Loss/Validation_Epoch', epoch_loss, epoch_index)
        writer.add_scalar('Metrics/R2_Score', val_r2, epoch_index)
        
    return epoch_loss, val_r2