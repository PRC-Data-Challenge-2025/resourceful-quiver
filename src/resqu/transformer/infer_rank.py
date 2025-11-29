# infer_rank.py
import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TabularTransformer
from dataset_infer import (
    FlightSequenceDatasetInference,
    flight_collate_fn_infer,
    TYPECODES,
    PHASE_VALUES,
)

# ---- CONFIG (match training) ----
data_dir = "rank_data_raw"  # <-- inference folder (no labels)
checkpoint_dir = "checkpoints"  # where you saved checkpoints
checkpoint_path = "checkpoints/checkpoint_epoch_0070.pt"  # set to a specific file or leave None to auto-pick latest
batch_size = 64

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
max_seq_len = 361

# Must match the model used during training:
MODEL_KWARGS = dict(
    type_emb_dim=32,
    d_model=256,
    nhead=16,
    num_layers=8,
    dim_feedforward=1024,
    dropout=0.15,
    max_seq_len=max_seq_len,
)

# Optional: normalization stats (same dict you used in training). If you used None, set to None here.
STATS = {
    "mean": {
        "CAS": 140.01529166159094,
        "TAS": 454.9053924817837,
        "airdist": 3397.329409108582,
        "altitude": 33434.47870860869,
        "compute_gs": 438.1911930008268,
        "compute_track": 186.2023623763514,
        "cumdist": 1767.1369651131813,
        "deltat": 1125.0190441144714,
        "distance_km": 3272.7376593896265,
        "ff_kgs_est": 1.2150995758031118,
        "fuel_kg": 1305.6280004880225,
        "groundspeed": 454.1061016738322,
        "idx": 61609.8734092951,
        "latitude": 28.39102746357758,
        "longitude": -32.548034576877676,
        "mach": 0.7827150290878517,
        "mass_est": 114451.58724705917,
        "specific_humidity": 0.0006663787092651272,
        "temperature": 229.2314779614455,
        "tow_est_kg": 135088.48518138775,
        "track": 174.6883780359047,
        "track_unwrapped": 161.32431085741953,
        "ts": 13745.19522822413,
        "u_component_of_wind": 12.171811299484274,
        "v_component_of_wind": -0.5444254829844769,
        "vertical_rate": -19.433975247357875,
        "x": -709720.1553305085,
        "y": 317649.885759743,
    },
    "std": {
        "CAS": 154.91858852257167,
        "TAS": 323.64203710270533,
        "airdist": 7028.8395394463,
        "altitude": 9523.96634155301,
        "compute_gs": 1050.805126840548,
        "compute_track": 100.86986478621897,
        "cumdist": 1602.0794105613377,
        "deltat": 925.2511058831888,
        "distance_km": 2967.0510683596085,
        "ff_kgs_est": 0.8404119682908922,
        "fuel_kg": 1526.9363191840866,
        "groundspeed": 325.1697646087668,
        "idx": 37752.574993560884,
        "latitude": 18.911142335498052,
        "longitude": 91.0382779082025,
        "mach": 0.5607426539325037,
        "mass_est": 57844.6075161093,
        "specific_humidity": 0.0024349654964718245,
        "temperature": 20.017739109524978,
        "tow_est_kg": 74275.61286143982,
        "track": 98.23456165218542,
        "track_unwrapped": 144.44895680098898,
        "ts": 10946.902583732199,
        "u_component_of_wind": 15.67457111943422,
        "v_component_of_wind": 11.497200504313675,
        "vertical_rate": 629.0541655173895,
        "x": 1735220.4786363742,
        "y": 1515446.4601253234,
    },
    "n_total_per_column": {},
}


def pick_latest_checkpoint():
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"{checkpoint_dir} not found")
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not files:
        raise FileNotFoundError("No checkpoints found")
    files.sort()
    return os.path.join(checkpoint_dir, files[-1])


def main():
    # Build dataset for rank_data_raw
    parquet_paths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".parquet")
    ]
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    ds = FlightSequenceDatasetInference(
        parquet_paths=parquet_paths,
        fixed_features=cfg["FIXED_FEATURES"],
        to_test_features=cfg["TO_TEST_FEATURES"],
        to_feed_end=cfg["TO_FEED_IN_THE_END"],
        stats=STATS,  # keep identical normalization to training
        max_seq_len=max_seq_len,  # match training
        step=10,  # match your training downsampling
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=flight_collate_fn_infer,
    )

    # Model
    seq_input_dim = (
        len(cfg["FIXED_FEATURES"]) + len(cfg["TO_TEST_FEATURES"]) + len(PHASE_VALUES)
    )
    end_input_dim = len(cfg["TO_FEED_IN_THE_END"])

    model = TabularTransformer(
        seq_input_dim=seq_input_dim,
        end_input_dim=end_input_dim,
        num_typecodes=len(TYPECODES),
        **MODEL_KWARGS,
    )

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # Load checkpoint
    if checkpoint_path:
        ckpt_path = checkpoint_path
    else:
        ckpt_path = pick_latest_checkpoint()

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Inference
    all_idx = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer"):
            seq_features = batch["seq_features"].to(device)
            end_features = batch["end_features"].to(device)
            typecode_id = batch["typecode_id"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            idx_label = batch["idx_label"]  # keep on CPU for writing

            preds = model(
                {
                    "seq_features": seq_features,
                    "end_features": end_features,
                    "typecode_id": typecode_id,
                    "attention_mask": attention_mask,
                }
            )  # [B]

            all_idx.extend(idx_label.tolist())
            all_pred.extend(preds.detach().cpu().tolist())

    # Save CSV: idx, predicted_fuel_kg
    out_df = (
        pd.DataFrame(
            {
                "idx": all_idx,
                "pred_fuel_kg": all_pred,
            }
        )
        .sort_values("idx")
        .reset_index(drop=True)
    )

    out_path = "predictions_rank.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
