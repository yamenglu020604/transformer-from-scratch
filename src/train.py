import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import yaml
import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import math

from model import Transformer
from dataset import create_dataloaders
from utils import plot_and_save_curves

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lr_scheduler(optimizer, warmup_steps, d_model):
    def lr_lambda(current_step):
        current_step += 1 # 1-based step
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return (d_model ** -0.5) * (current_step ** -0.5) * (warmup_steps ** 0.5)
    
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, config):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")

    for src, tgt_in, tgt_out in pbar:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        
        optimizer.zero_grad()
        
        pad_idx = dataloader.dataset.src_tokenizer.token_to_id('[PAD]')
        src_mask, tgt_mask = Transformer.create_masks(src, tgt_in, pad_idx, device)
        
        output = model(src, tgt_in, src_mask, tgt_mask)
        
        loss = criterion(output.view(-1, output.shape[-1]), tgt_out.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_thresh'])
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for src, tgt_in, tgt_out in pbar:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            
            pad_idx = dataloader.dataset.src_tokenizer.token_to_id('[PAD]')
            src_mask, tgt_mask = Transformer.create_masks(src, tgt_in, pad_idx, device)

            output = model(src, tgt_in, src_mask, tgt_mask)
            
            loss = criterion(output.view(-1, output.shape[-1]), tgt_out.view(-1))
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
    return total_loss / len(dataloader)


def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, src_tokenizer, tgt_tokenizer = create_dataloaders(config)
    
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    pad_idx = src_tokenizer.token_to_id('[PAD]')

    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        config['d_model'],
        config['n_heads'],
        config['n_layers'],
        config['d_ff'],
        config['max_seq_len'],
        config['dropout']
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_lr_scheduler(optimizer, config['lr_warmup_steps'], config['d_model'])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=config['label_smoothing'])

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{config['epochs']} ---")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, device, config)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"Model saved to {config['model_save_path']}")

    plot_and_save_curves(train_losses, val_losses, config['results_dir'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model from scratch.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    main(args.config)