import argparse
import torch
import os
from datetime import datetime

from data import get_dataloaders
from model import GPT, GPTConfig
from lora import LoRA, LoRAConfig
from trainer import train_model, evaluate_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train a local version of GPT2")

    #general args
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--block_size", type=int, default=128, help="Context window size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of full passes over data")

    #lora args
    parser.add_argument("--use_lora", action="store_true", help="Add this flag to enable Low Rank Adaptation")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank for LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha scaling parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float,  )

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
        model = LoRA.inject_lora(pretrained_model, lora_config= LoRAConfig(
            rank = args.lora_rank,
            alpha = args.lora_alpha,
            dropout= args.lora_dropout
            ))
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, 
                                      lr = 3e-4, 
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
    if not os.path.exists("./models/"):
        os.mkdir("./models")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(ft_model,f"./models/model_ft_{timestamp}")

if __name__ == "__main__":
    main()
