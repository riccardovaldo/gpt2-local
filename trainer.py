import torch
from torch.utils.data import DataLoader
import argparse
import os
from datetime import datetime
from data import get_dataloaders
from model import GPT
from lora import LoRA, LoRAConfig

@torch.no_grad()
def evaluate_loss(model, val_loader, device):
    "Evaluate validation loss after every epoch"

    model.eval()
    total_loss = 0.0

    for x,y in val_loader:
        x,y  = x.to(device), y.to(device)
        _, loss = model(x, targets = y)
        total_loss += loss.item()
    model.train()
    return total_loss/len(val_loader)

def train_model(model, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim,
        epochs: int, 
        device: str):
    
    model = model.to(device)
    model.train()

    print("Starting training...")

    for epoch in range(epochs):
        total_train_loss = 0.0
        for step, (x,y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits, loss = model(x, targets = y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            total_train_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step} | Train Loss: {loss.item():.4f}")
    
        avg_train_loss = total_train_loss/len(train_loader)
        avg_val_loss = evaluate_loss(model, val_loader, device)
        print(f"===== Epoch {epoch + 1} completed =====")
        print(f"Train loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
    
    print("Training complete!")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a local version of GPT2")

    #general args
    parser.add_argument("--batch_size", type = int, default = 4, help = "Batch size per step")
    parser.add_argument("--block_size", type = int, default = 128, help = "Context window size")
    parser.add_argument("--lr", type = float, default = 3e-4, help = "Learning rate")
    parser.add_argument("--epochs", type = int, default = 3, help = "Number of full passes over data")

    #lora args
    parser.add_argument("--use_lora", action = "store_true", help = "Add this flag to enable Low Rank Adaptation")
    parser.add_argument("--lora_rank", type = int, default = 4, help = "Rank for LoRA matrices")
    parser.add_argument("--lora_alpha", type = int, default = 16, help = "Alpha scaling parameter for LoRA")
    parser.add_argument("--lora_dropout", type = float, default = 0.05, help = "Dropout percentage applied to the lora layers."   )

    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    train, val = get_dataloaders(args.batch_size, args.block_size)

    print("Initializing the model...")
    pretrained_model = GPT.from_pretrained()

    if args.use_lora:
        print("Injecting LoRA modules inside the model")
        model = LoRA.inject_lora(pretrained_model, config= LoRAConfig(
            rank = args.lora_rank,
            alpha = args.lora_alpha,
            dropout= args.lora_dropout
            ))
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, 
                                      lr = args.lr, 
                                      weight_decay = 0.01) #optimizer defaul when using lora
    else:
        print("Training the entire model without LoRA")
        model = pretrained_model 
        optimizer = GPT.configure_optimizer() #optimizer w/ weight decay when performing classic ft
    
    model = model.to(device)

    #training loop 
    ft_model = train_model(model = model,
                           train_loader = train,
                           val_loader = val,
                           optimizer = optimizer ,
                           epochs = args.epochs,
                           device = device)
    
    os.makedirs("./models/", exist_ok = True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.use_lora:
        state_to_save = {k:v for k,v in model.state_dict().item() if k.__contains__("lora")}
        save_path = f"./models/model_loraft_{timestamp}.pt"
    else: 
        state_to_save = model.state_dict()
        save_path = f"./models/model_ft_{timestamp}.pt"
        
    torch.save(state_to_save, save_path)
    print(f"Model weights saved successfully to {save_path}.")

if __name__ == "__main__":
    main()