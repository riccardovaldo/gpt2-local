from dataclasses import dataclass
import torch
import math
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
        return 0.5*x*(1.0 + torch.tanh(math.sqrt(2/torch.pi)*(x + 0.047715 * torch.pow(x, 3.0))))
    
class MLP(nn.Module):
    """Implementation of the MLP"""
    def __init__(self, config: GPTConfig):
        super().__init__()

        #Mantaining the layer names used in the original implementation to smooth out the process of preloading original weights
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.act = MGelu()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj._is_residual = True
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    """Implementation of the Attention Module"""

    def __init__(self, config: GPTConfig):
        super().__init__()

        #check if the dmodel adapts to the n. of heads which are acting only on subsection of the dimensions
        assert config.n_embed % config.n_head == 0 

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj._is_residual = True

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        #automatically moves the mask to a device if the model is sent to the same device
        #if not I'd have to move the mask manually in the forward pass making it less efficient
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))\
                             .view(1,1,config.block_size, config.block_size))

        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x):
        B, T, C = x.size()
        #splitting on the 2nd dimension because rn I do have a giant M = (B,T,3C)
        q, k, v = self.c_attn(x).split(self.n_embed, dim = 2)

        #introduce heads in the computations
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B,nH,T,nD)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B,nH,T,nD)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B,nH,T,nD)

        att = q@k.transpose(-2,-1) * (1.0 / math.sqrt(k.size(-1)))
        #casual mask application
        #:T is used for adaptability to the dataset being used
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf")) 
        #softmax across the last dimension, which means horizontally on the columns
        att = F.softmax(att, dim = -1)
        att = self.attn_dropout(att)

        y = att @ v #(B,nH,T,T) @ (B,nH,T,nD) = B,nH,T,nD)
        y = y.transpose(1,2).contiguous().view(B,T,C) #recompose heads

        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y 

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

        
        
class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embed),
                wpe = nn.Embedding(config.block_size, config.n_embed),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embed)))
        
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialize weights wrt to GPT2 paper"""

        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, '_is_residual'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None: 
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, targets = None):
        B, T = idx.size()
        device = idx.device

        #Initialize embeddings
        pos  = torch.arange(0, T, device = device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        #Run through blocks
        for block in self.transformer.h:
            x = block(x)

        #Final step
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(dim=-1)), targets.view(-1), ignore_index=-1)
            #ignore index is used to avoid the loss computation when sentences are shorter than block size
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, do_sample = False, top_k = None, temperature = 1.0):

        for t in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))

                #set the values less then topK to -inf to exclude them from the softmax
                logits[logits < values[:, [-1]]] = float("-Inf")
            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

    @classmethod     
    def from_pretrained(cls):
        """
        Loader for the GPT2 original weights from Hugging Face
        """
        from transformers import GPT2LMHeadModel
        print("Loading the official GPT2 (124M) weights...")

        config = GPTConfig
        model = GPT(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()
        
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # we do not import the lower triangular mask in casual attention
        #needs to be transposed because of the openai conv1d
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        sd_keys = [k for k in sd.keys() if not k.endswith('attn.bias')]
        assert len(keys) == len(sd_keys), f"mismatched keys: {len(keys)} != {len(sd_keys)}"

        for k in keys:
            if any(k.endswith(w) for w in transposed):
                assert sd[k].shape == sd_hf[k].shape[::-1]  
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model

                