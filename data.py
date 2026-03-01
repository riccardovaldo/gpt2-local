import os
import torch
from torch.utils.data import Dataset, DataLoader
from model import GPTConfig
from urllib import request
import tiktoken

class ShakespearData(Dataset):
    def __init__(self, block_size):
        
        self.block_size = block_size
        self.data_path = "data.txt"

        if not os.path.exists(self.data_path):
            print("Downloading Shakespear dataset...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            request.urlretrieve(url, self.data_path)
        
        with open(self.data_path, 'r', encoding = 'utf-8') as file:
            raw_data = file.read()

        print("Tokenizing data...")

        encoder = tiktoken.get_encoding('gpt2')
        self.tokens = encoder.encode(raw_data)

        print(f"The dataset has {len(self.tokens)} tokens.")
    
    def __len__(self,):
        #avoid going over the end of the data
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        item = self.tokens[idx : idx + self.block_size + 1]

        x = torch.tensor(item[:-1], dtype=torch.long)
        y = torch.tensor(item[1:], dtype=torch.long)

        return x,y

def get_dataloaders(batch_size: int = 4, block_size: int = 1024):

    dataset = ShakespearData(block_size = block_size)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = batch_size,
                              shuffle= True,
                              drop_last = True
                            )
    val_loader = DataLoader(dataset = val_dataset,
                              batch_size = batch_size,
                              shuffle= False,
                              drop_last = True
                            )



    
        
    


