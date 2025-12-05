import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils import compute_metrics

def evaluate_model(model, loader, y_scaler, device, writer=None, step=None):
    model.eval()
    preds_list = []
    targets_list = []
    
    pbar = tqdm(loader, desc="Evaluating", leave=True, unit="batch")
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.numpy()) 
            
    y_pred_scaled = np.vstack(preds_list)
    y_true_scaled = np.vstack(targets_list)
    
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
    y_true_original = y_scaler.inverse_transform(y_true_scaled)
    
    metrics = compute_metrics(y_true_original, y_pred_original)
    
    if writer is not None and step is not None:
        # Visual Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        limit = 500 
        flat_true = y_true_original[:limit].flatten()
        flat_pred = y_pred_original[:limit].flatten()
        
        ax.scatter(flat_true, flat_pred, alpha=0.5, s=10) # s=10 for smaller dots
        
        min_val = min(flat_true.min(), flat_pred.min())
        max_val = max(flat_true.max(), flat_pred.max())
        
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        ax.set_xlabel("Actual dq (rad)")
        ax.set_ylabel("Predicted dq (rad)")
        ax.legend()
        ax.set_title(f"Epoch {step}: Preds vs Actual")
        
        writer.add_figure("Evaluation/Plots", fig, step)
        plt.close(fig)
    
    return metrics, y_true_original, y_pred_original