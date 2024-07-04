import torch
from torch import nn 
class ScaledTanh(nn.Module):
    def __init__(self, scale=0.3):
        super(ScaledTanh, self).__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * torch.tanh(x)
class Actor(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,num_layers):
        super().__init__()
        self.l1 = nn.LSTM(input_dim,hidden_dim,num_layers,batch_first = True)
        self.l2 = nn.Linear(hidden_dim,output_dim)
        self.stanh = ScaledTanh(scale = 0.3)
    def forward(self,x):
        _, (lstm_output,_) = self.l1(x)
        lstm_output = lstm_output.squeeze(0)  
        x = self.stanh(self.l2(lstm_output)) 
        
        return x


class Critic(nn.Module):
    def __init__(self,input_dim_obs,hidden_dim,output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim_obs,hidden_dim,batch_first=True)
        self.l1 = nn.Linear(hidden_dim,output_dim)
        self.l2 = nn.Linear(2*output_dim,500)
        
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(500,1000)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(1000,1000)
        self.relu4 = nn.ReLU()
        self.l5 = nn.Linear(1000,1000)
        self.relu5 = nn.ReLU()
        self.l6 = nn.Linear(1000,1000)
        self.relu6 = nn.ReLU()
        self.l7 = nn.Linear(1000,500)
        self.relu7 = nn.ReLU()
        self.l8 = nn.Linear(500,200)
        self.relu8 = nn.ReLU()
        self.l9 = nn.Linear(200,1)
        
        
    def forward(self,obs,co):
        _,(hidden,_) = self.lstm(obs)
        
        lstm_output = hidden.squeeze(0)
        lstm_output = torch.tanh(self.l1(lstm_output))
        combined = torch.cat((lstm_output,co),dim = 1)

        x = self.l2(combined)
        x = self.relu2(x)
        x = self.l3(x)
        x = self.relu3(x)
        x = self.l4(x)
        x = self.relu4(x)
        x = self.l5(x)
        x = self.relu5(x)
        x = self.l6(x)
        x = self.relu6(x)
        x = self.l7(x)
        x = self.relu7(x)
        x = self.l8(x)
        x = self.relu8(x)
        x = self.l9(x)
        return x
    

import json

def save_model(model, path):
    # Save the model weights
    weights_path = f"{path}_weights.pth"
    torch.save(model.state_dict(), weights_path)
    
    # Save the model architecture
    arch_path = f"{path}_arch.json"
    model_arch = {
        "class": model.__class__.__name__,
        "args": {},  
        "state_dict_path": weights_path
    }
    with open(arch_path, 'w') as f:
        json.dump(model_arch, f)

    print(f"Model weights saved to {weights_path}")
    print(f"Model architecture saved to {arch_path}")

def load_model(path, model):
    # Load the architecture file
    arch_path = f"{path}_arch.json"
    with open(arch_path, 'r') as f:
        model_arch = json.load(f)
    
    # Load the weights
    model.load_state_dict(torch.load(model_arch["state_dict_path"]))
    model.eval()  # Set the model to evaluation mode

    print(f"Model loaded from {model_arch['state_dict_path']}")
    return model

