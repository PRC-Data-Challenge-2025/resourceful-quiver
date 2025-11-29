import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from dataset import FlightSequenceDataset, flight_collate_fn, TYPECODES, PHASE_VALUES
from model import TabularTransformer

from resqu import data
# --------------------
# Config
# --------------------

batch_size = 64
num_epochs = 400
base_lr = 5e-5
weight_decay = 1e-3
train_ratio = 0.7
data_dir = data.data_dir / "train_data_raw"
log_path = data.log_dir / "training_log.json"
checkpoint_dir = data.checkpoint_dir

cfg = {
    "FIXED_FEATURES": ["latitude", "longitude", "altitude", "ts"],
    "TO_TEST_FEATURES": [
        "vertical_rate",
        "mach",
        "TAS",
        "CAS",
        "cumdist",
        "ff_kgs_est",
    ],
    "TO_FEED_IN_THE_END": ["tow_est_kg", "deltat"],
}

max_seq_len = 361  # CLS + up to 721 steps or whatever you chose


# --------------------
# LR Scheduler (warmup + linear decay)
# --------------------

def get_linear_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup, then linear decay to 0."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)


# --------------------
# Loss
# --------------------

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2)
        return torch.sqrt(mse + self.eps)


# --------------------
# Train / Eval loops
# --------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        seq_features = batch["seq_features"].to(device)
        end_features = batch["end_features"].to(device)
        typecode_id = batch["typecode_id"].to(device)
        fuel_kg_label = batch["fuel_kg_label"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        preds = model({
            "seq_features": seq_features,
            "end_features": end_features,
            "typecode_id": typecode_id,
            "attention_mask": attention_mask,
        })  # [B]

        loss = loss_fn(preds, fuel_kg_label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_loss = loss.item()
        running_loss += batch_loss * seq_features.size(0)

        pbar.set_postfix({
            "true_RMSE": f"{batch_loss:.4f}",
        })

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for batch in pbar:
            seq_features = batch["seq_features"].to(device)
            end_features = batch["end_features"].to(device)
            typecode_id = batch["typecode_id"].to(device)
            fuel_kg_label = batch["fuel_kg_label"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            preds = model({
                "seq_features": seq_features,
                "end_features": end_features,
                "typecode_id": typecode_id,
                "attention_mask": attention_mask,
            })

            loss = loss_fn(preds, fuel_kg_label)
            batch_loss = loss.item()
            running_loss += batch_loss * seq_features.size(0)

            pbar.set_postfix({
                "true_RMSE": f"{batch_loss:.4f}"
            })

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


# --------------------
# Checkpoint utils
# --------------------

def save_checkpoint(epoch, model, optimizer, scheduler, train_rmse, val_rmse):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_RMSE": train_rmse,
        "val_RMSE": val_rmse,
        "config": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "base_lr": base_lr,
            "weight_decay": weight_decay,
            "train_ratio": train_ratio,
            "cfg": cfg,
            "max_seq_len": max_seq_len,
        },
    }
    # path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    path = checkpoint_dir / f"checkopint_epoch_{epoch:04d}.pt"
    torch.save(ckpt, path)


# --------------------
# Main
# --------------------

def main():
    # Dataset paths
    parquet_paths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".parquet")
    ]

    # Dataset
    dataset = FlightSequenceDataset(
        parquet_paths=parquet_paths,
        fixed_features=cfg["FIXED_FEATURES"],
        to_test_features=cfg["TO_TEST_FEATURES"],
        to_feed_end=cfg["TO_FEED_IN_THE_END"],
        max_seq_len = max_seq_len
        # stats=stats_dict,  # plug in if you have normalization
    )

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=flight_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=flight_collate_fn,
    )

    print(f"Total sequences: {n_total} | Train: {n_train} | Val: {n_val}")

    # Model
    seq_input_dim = (
        len(cfg["FIXED_FEATURES"])
        + len(cfg["TO_TEST_FEATURES"])
        + len(PHASE_VALUES)
    )
    end_input_dim = len(cfg["TO_FEED_IN_THE_END"])

    model = TabularTransformer(
        seq_input_dim=seq_input_dim,
        end_input_dim=end_input_dim,
        num_typecodes=len(TYPECODES),
        type_emb_dim=32,
        d_model=256,
        nhead=16,
        num_layers=8,
        dim_feedforward=1024,
        dropout=0.15,
        max_seq_len=max_seq_len,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )
    loss_fn = RMSELoss()

    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_scheduler_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Load logs if exist
    logs = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_rmse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scheduler,
        )
        val_rmse = evaluate(model, val_loader, loss_fn, device)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:04d} | "
            f"train RMSE: {train_rmse:.4f} | "
            f"val RMSE: {val_rmse:.4f} | "
            f"lr: {current_lr:.6e}"
        )

        # Save checkpoint every 20 epochs
        if epoch % 10 == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, train_rmse, val_rmse)

        # Log
        record = {
            "epoch": epoch,
            "train_RMSE": train_rmse,
            "val_RMSE": val_rmse,
            "lr": current_lr,
            "batch_size": batch_size,
        }
        logs.append(record)
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=4)


if __name__ == "__main__":
    main()
