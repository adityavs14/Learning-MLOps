import torch
import torch.nn as nn

device = "cpu"


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        
        self.inp_layer = nn.Linear(input_dim, hidden_dims[0]).to(device)
        self.out_layer = nn.Linear(hidden_dims[-1], output_dim).to(device)
        
        self.layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1])
                                     for i in range(0,len(hidden_dims) - 1)]).to(device)
        
        self.relu = nn.ReLU().to(device)
        self.act = nn.Sigmoid().to(device)
        
        
    def forward(self, x):
        x.to(device)
        
        x = self.relu(self.inp_layer(x))
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
            
        out = self.out_layer(x)
        # out = self.act(out)
        
        return out        
        