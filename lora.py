import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import List

@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: int = 16
    dropout: float = 0.05
    #layer injections (default factory creates a list every time the class is initalized on different mem positions)
    target_layers: List[str] = field(default_factory= lambda: ["c_attn", "c_proj", "c_fc"])

class LoRA(nn.Module):
    def __init__(self, original_layer: nn.Linear, config: LoRAConfig = LoRAConfig()):
        super().__init__()

        #save and freeze the original layer
        self.original_layer = original_layer

        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        #lora params
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = self.alpha / self.rank

        #lora layer
        self.lora_b = nn.Linear(self.in_features, self.rank, bias = False)
        self.lora_a = nn.Linear(self.rank, self.out_features, bias = False)
        self.dropout = nn.Dropout(p= config.dropout)

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
        w_x = self.original_layer(x)
        a_x = self.lora_a(self.dropout(x))
        b_x = self.lora_b(a_x)
        y = w_x + self.lora_b(a_x) * self.scaling
        return y

    @classmethod
    def inject_lora(cls, model, config: LoRAConfig):
        """
        Take a GPT class model as an arg and inject lora layers in place of any linear layer but lm_head 
        """

        for mn, m in model.named_modules():
            for cn, c in m.named_children():
                
                if isinstance(c, nn.Linear) and any(target in cn for target in config.target_layers):

                    lora_layer = cls(c)
                    setattr(m, cn, lora_layer)
        
        return model
        




        





    