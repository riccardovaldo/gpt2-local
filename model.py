from dataclasses import dataclass
import torch
import math
import numpy
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    # --- Dropout ---
    emb_pdrop: float = 0.1 #for embeddings
    attn_pdrop: float = 0.1 #after the casual attention 
    resid_pdrop: float = 0.1 #for MLP and attention output


class MGelu(nn.Module):
    """Modified Gelu implementation to adapt the GPT 2 official GELU"""

    def forward(self, x):
        return 0.5*x*(1.0 + torch.tanh(torch.sqrt(2/torch.pi)*(x + 0.047715 * torch.pow(x, 3.0))))
    
class MLP(nn.Module):
    """Implementation of the MLP"""
    def __init__(self, config: GPTConfig):
        super().__init()

        #Mantaining the layer names used in the original implementation to smooth out the process of preloading original weights
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.act = MGelu()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    """Implementation of the Attention Module"""

    def __init__(self, config):
        super().__init__()

        #check if the dmodel adapts to the n. of heads which are acting only on subsection of the dimensions
        assert config.n_embed % config.n_head == 0 

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        #check for the casual masking 

        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embed, dim = 2)
        


