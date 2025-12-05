import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class RobotDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def filter_data(df, input_cols, output_cols):
    """
    Applies constraints suggested.
    """
    initial_len = len(df)
    
    # Extract values once
    dq_values = df[output_cols].values
    
    # Constraint 1: Small Displacements (|dq| < 10 degrees)
    limit_rad = np.deg2rad(10.0) 
    mask_upper = np.all(np.abs(dq_values) < limit_rad, axis=1)
    
    # Constraint 2: Remove static/noise data (sum |dq| > epsilon)
    mask_lower = np.sum(np.abs(dq_values), axis=1) > 1e-4
    
    # Constraint 3: Joint Limits
    mask_joints = np.ones(len(df), dtype=bool)
    for col in input_cols:
        if 'q' in col:
            # Joint 1: -pi to +pi
            if col == 'q1':
                is_valid = (df[col] >= -np.pi) & (df[col] <= np.pi)
            # Other Joints: -1.8 to +1.8
            else:
                is_valid = (df[col] >= -1.8) & (df[col] <= 1.8)
            
            mask_joints = mask_joints & is_valid

    # Combine all masks
    final_mask = mask_upper & mask_lower & mask_joints
    df_filtered = df[final_mask].reset_index(drop=True)
    
    print(f"Data Filtering: Removed {initial_len - len(df_filtered)} rows "
          f"({(initial_len - len(df_filtered))/initial_len:.1%}).")
    
    return df_filtered

def load_and_process_data(csv_paths, input_cols, output_cols, val_split=0.2, seed=42):
    """
    Loads, cleans, splits, and scales the data.
    """
    # 1. Load Data
    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    
    # Check for missing columns
    missing = [c for c in input_cols + output_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 2. Apply constraints suggested
    df = filter_data(df, input_cols, output_cols)

    # 3. Extract Values
    X_raw = df[input_cols].values
    y_raw = df[output_cols].values

    # 4. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=val_split, random_state=seed
    )

    # 5. Scale
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)

    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val)

    # 6. Dataset Creation
    train_dataset = RobotDataset(X_train_scaled, y_train_scaled)
    val_dataset = RobotDataset(X_val_scaled, y_val_scaled)

    return train_dataset, val_dataset, x_scaler, y_scaler

def load_test_data(csv_paths, input_cols, output_cols, x_scaler, y_scaler):
    """
    Loads test data and applies the SCALERS fitted on the training set.
    """
    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    
    # Check columns
    missing = [c for c in input_cols + output_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Test Set: {missing}")

    # 1. Apply same filtering as training
    df = filter_data(df, input_cols, output_cols)
    
    X_raw = df[input_cols].values
    y_raw = df[output_cols].values
    
    # 2. Apply same scaling as training
    X_scaled = x_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(y_raw)
    
    # 3. Create Dataset
    dataset = RobotDataset(X_scaled, y_scaled)
    
    return dataset