import torch
import yaml
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.data.dataset import How2SignDataset
from src.model.model import SignLanguageTransformer


def set_seed(seed: int):
    """Sets random seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Loads the YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, optimizer, criterion, device) -> float:
    """Runs one full training epoch and returns the average loss."""
    model.train()
    total_loss = 0

    for sequence, labels, attention_mask, valid_frames in dataloader:
        sequence  = sequence.to(device)
        labels    = labels.to(device)
        attention_mask = attention_mask.to(device)

        src_key_padding_mask = ~valid_frames
        src_key_padding_mask = src_key_padding_mask.to(device)

        tgt_input  = labels[:, :-1]
        tgt_output = labels[:, 1:]

        tgt_key_padding_mask = (attention_mask[:, :-1] == 0)

        optimizer.zero_grad()
        output = model(
            sequence,
            tgt_input,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def val_epoch(model, dataloader, criterion, device) -> float:
    """Runs one full validation epoch and returns the average loss."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for sequence, labels, attention_mask, valid_frames in dataloader:
            sequence       = sequence.to(device)
            labels         = labels.to(device)
            attention_mask = attention_mask.to(device)

            src_key_padding_mask = ~valid_frames
            src_key_padding_mask = src_key_padding_mask.to(device)

            tgt_input  = labels[:, :-1]
            tgt_output = labels[:, 1:]
            tgt_key_padding_mask = (attention_mask[:, :-1] == 0)

            output = model(
                sequence,
                tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    config = load_config()
    set_seed(config['training']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = How2SignDataset(config['data']['processed_dir'], 'train')
    val_dataset   = How2SignDataset(config['data']['processed_dir'], 'val')

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['training']['batch_size'])

    model     = SignLanguageTransformer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config['training']['label_smoothing'])

    patience = config['training']['early_stopping_patience']
    patience_counter = 0

    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = val_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config['training']['epochs']} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_dir / 'best_model.pt'
            )

            print("  → checkpoint saved")

        else:
            patience_counter += 1
            print(f"  → patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

if __name__ == '__main__':
    main()