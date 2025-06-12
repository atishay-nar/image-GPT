# imports
import argparse
import yaml
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from image_gpt import ImageGPT 
from tokenize_data import TokenizedData

# set device
DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
        )

def train(cfg):
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)

    # load data
    train_data = TokenizedData(cfg)
    train_loader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=True)

    # define model and optimizer
    model = ImageGPT(cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR) 

    #put model in training mode
    model.train()
    criterion = nn.CrossEntropyLoss()
    epochs = cfg.EPOCHS

    # TO-DO: add a scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs*len(train_loader))  # cosine annealing scheduler

    

    # training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = batch.long().to(DEVICE) # batch: (B, seq_len)
            inputs = batch[:, :-1] # first seq_len-1 tokens for each image
            targets = batch[:, 1:] # last seq_len-1 tokens for each image


            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) # calculate loss based on logits and targets
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        # TO-DO: add scheduler step
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0 or (epoch + 1) == epochs:
            save_path = os.path.join(cfg.MODEL_SAVE_DIR, f"image_gpt_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)
    train(cfg)