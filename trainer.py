import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

def train(model, 
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
            logits, loss = model(x, targets = y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            total_train_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step} | Train Loss: {loss.item():.4f}")
    
        avg_train_loss = total_train_loss/len(train_loader)
        avg_val_loss = evaluate_loss(model, val_loader, device)
        print("===== Epoch {epoch + 1} completed =====")
        print(f"Train loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
    
    print("Training complete!")
    return model
