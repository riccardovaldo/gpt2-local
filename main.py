import argparse
import torch
from torch.optim import AdamW

from data import get_dataloaders
from model import GPT, GPTConfig
from lora import LoRA, LoRAConfig

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

    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    train, val = get_dataloaders(args.batch_size, args.block_size)

    print("Initializing the model...")
    model = GPT.from_pretrained()

    if args.use_lora:
        print("Injecting LoRA modules inside the model")
        model = LoRA.inject_lora(model, LoRAConfig)
    else:
        print("Training the entire model without LoRA")
    
    model = model.to(device)

    #training loop

    


if __name__ == "__main__":
    main()
