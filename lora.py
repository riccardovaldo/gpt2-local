import torch
import torch.nn as nn
from torch.nn import functional as F

class LoRA(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: int = 16):
        super().__init__()

        #save and freeze the original layer
        self.original_layer = original_layer
        
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        #lora params
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank

        #lora layer
        self.lora_b = nn.Linear(self.in_features, self.rank, bias = False)
        self.lora_a = nn.Linear(self.rank, self.out_features, bias = False)

        #initialiaze only lora A and B
        self.reset_params()
    
    def reset_params(self):
        """
        Initialize self.lora_a and self.lora_b
        """
        #in the original paper A is initilized with a Gaussian but here we follow the peft library form HF
        nn.init.kaiming_uniform_(self.lora_a.weight) 
        nn.init.zeros_(self.lora_b.weight)
    
    def forward(self, x):

    




    