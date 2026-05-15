import torch
import yaml
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import How2SignDataset
from src.model.model import SignLanguageTransformer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def train_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0

    for sequence, labels, attention_mask, valid_frames in dataloader:
        sequence       = sequence.to(device)
        labels         = labels.to(device)
        attention_mask = attention_mask.to(device)

        src_key_padding_mask = (~valid_frames).to(device)

        tgt_input  = labels[:, :-1]
        tgt_output = labels[:, 1:]
        tgt_key_padding_mask = (attention_mask[:, :-1] == 0)

        optimizer.zero_grad()
        output = model(
            sequence, tgt_input,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        output     = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def val_epoch(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for sequence, labels, attention_mask, valid_frames in dataloader:
            sequence       = sequence.to(device)
            labels         = labels.to(device)
            attention_mask = attention_mask.to(device)

            src_key_padding_mask = (~valid_frames).to(device)

            tgt_input  = labels[:, :-1]
            tgt_output = labels[:, 1:]
            tgt_key_padding_mask = (attention_mask[:, :-1] == 0)

            output = model(
                sequence, tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            output     = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    config = load_config()
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 60)

    vocab_path = config["data"]["vocab_path"]

    # Get max output length from config to synchronize lengths
    max_out_len = config["model"].get("max_output_length", 30)

    train_dataset = How2SignDataset(
        config["data"]["processed_dir"], 
        "train", 
        vocab_path=vocab_path,
        max_output_length=max_out_len
    )
    val_dataset = How2SignDataset(
        config["data"]["processed_dir"], 
        "val",   
        vocab_path=vocab_path,
        max_output_length=max_out_len
    )
    
    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Vocab size    : {config['model']['vocab_size']}")
    print("=" * 60)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    model = SignLanguageTransformer(config).to(device)

    params = count_parameters(model)
    print(f"Total parameters    : {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}  (backbone frozen for first {config['model'].get('freeze_epochs', 10)} epochs)")
    print("=" * 60)

    freeze_epochs = config["model"].get("freeze_epochs", 10)

    # Two param groups: backbone (low lr) and rest (normal lr)
    backbone_params = list(model.encoder.dstformer.parameters())
    other_params    = [p for p in model.parameters() if not any(p is bp for bp in backbone_params)]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params,    "lr": config["training"]["learning_rate"]},
            {"params": backbone_params, "lr": config["training"]["learning_rate"] * 0.1},
        ],
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=config["training"]["label_smoothing"],
    )

    patience         = config["training"]["early_stopping_patience"]
    patience_counter = 0
    best_val_loss    = float("inf")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    max_epochs = config["training"]["epochs"]

    for epoch in range(max_epochs):
        # Notify encoder of current epoch (handles freeze/unfreeze)
        model.encoder.set_epoch(epoch)

        # Update trainable count after unfreeze
        if epoch == freeze_epochs:
            params = count_parameters(model)
            print(f"  [info] Trainable parameters now: {params['trainable']:,}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = val_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Detailed log line
        phase = "freeze" if epoch < freeze_epochs else "finetune"
        print(
            f"Epoch {epoch+1:03d}/{max_epochs} [{phase}] | "
            f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
            f"lr: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print("  → checkpoint saved")
        else:
            patience_counter += 1
            print(f"  → patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

    print("=" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()