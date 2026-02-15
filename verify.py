import torch 
import tiktoken
from model import GPT

model = GPT.from_pretrained()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

enc = tiktoken.get_encoding('gpt2')
txt = """Q: What is the capital of Germany?
A: Berlin
Q: What is the capital of Italy?
A: Rome
Q: What is the capital of France?
"""    
tokens = torch.tensor(enc.encode(txt), dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    complete = model.generate(idx = tokens, 
                              max_new_tokens = 20,
                              do_sample= True,
                              top_k = 10,
                              temperature = 1.0)

response = enc.decode(complete[0].tolist())
print(response)
