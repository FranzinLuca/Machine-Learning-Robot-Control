import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation_fn, dropout_rate, use_residual):
        super(ResidualBlock, self).__init__()
        self.use_residual = use_residual
        
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.shortcut = nn.Identity()
        if self.use_residual:
            if in_features != out_features:
                self.shortcut = nn.Linear(in_features, out_features)
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        # Main processing
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Add skip connection
        if self.use_residual:
            out += self.shortcut(x)
            
        return out

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], 
                 dropout_rate=0.0, activation='tanh', use_residual=False):
        super(DynamicMLP, self).__init__()
        
        # Select Activation Class
        if activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        layers = []
        in_features = input_dim
        
        # Hidden Layers
        for hidden_dim in hidden_layers:
            layers.append(ResidualBlock(
                in_features=in_features,
                out_features=hidden_dim,
                activation_fn=act_fn,
                dropout_rate=dropout_rate,
                use_residual=use_residual
            ))
            in_features = hidden_dim
            
        # Output layer
        self.output_layer = nn.Linear(in_features, output_dim)
        
        # Wrap hidden layers in Sequential
        self.hidden_net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.hidden_net(x)
        x = self.output_layer(x)
        return x