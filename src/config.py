import torch

class Config:
    # --- Paths ---
    DATA_DIR = "data"
    MODELS_DIR = "models"
    PLOTS_DIR = "plots"
    
    SEEDS = [42, 101, 420, 2024] 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ROBOT_CHOICE = ['Reacher3', 'Reacher4', 'Reacher6']

    # Data specifics
    INPUTS_REACHER3 = ['x', 'y', 'q1', 'q2', 'q3', 'dx', 'dy']
    OUTPUTS_REACHER3 = ['dq1', 'dq2', 'dq3']
    
    INPUTS_REACHER4 = ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4', 'dx', 'dy', 'dz']
    OUTPUTS_REACHER4 = ['dq1', 'dq2', 'dq3', 'dq4']
    
    INPUTS_REACHER6 = ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'dx', 'dy', 'dz']
    OUTPUTS_REACHER6 = ['dq1', 'dq2', 'dq3', 'dq4', 'dq5', 'dq6']
    
    # Training hyperparameters
    BATCH_SIZE = 256
    EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 30
    
    # Optuna hyperparameter ranges
    EPOCHS_OPTUNA = 50  # Reduced epochs for tuning
    NUM_LAYERS_RANGE = (6, 12)
    HIDDEN_UNITS_OPTIONS = [128, 256, 512, 1024, 2048]
    DROPOUT_RANGE = (0.0, 0.5)
    LR_RANGE = (5e-4, 1e-2)
    ACTIVATION_OPTIONS = ['relu', 'leaky_relu']
    RUN_OPTIMIZATION = False
    NUM_TRIALS = 100
    LOAD_FROM_OPTUNA = False

    # Best hyperparameters (enhanced optimizer work)
    def best_hyperparameters(robot_name):
        if robot_name == 'Reacher3':
            return {
                'num_layers': 4,
                'hidden_units': [2048, 1024, 512, 64],
                'dropout': 0.01,
                'learning_rate': 5.4 * 1e-5,
                'activation': 'leaky_relu',
                'use_residuals': False
            }
        elif robot_name == 'Reacher4':
            return {
                'num_layers': 3,
                'hidden_units': [1024, 2048, 1024],
                'dropout': 0.2,
                'learning_rate': 7 * 1e-5,
                'activation': 'relu',
                'use_residuals': False
            }
        elif robot_name == 'Reacher6':
            return {
                'num_layers': 10,
                'hidden_units': [2048, 2048, 1024, 1024, 1024, 1024, 512, 512, 256, 128],
                'dropout': 0.17,
                'learning_rate': 5.2 * 1e-5,
                'activation': 'leaky_relu',
                'use_residuals': True
            }
        else:
            raise ValueError("Unknown robot name provided.")