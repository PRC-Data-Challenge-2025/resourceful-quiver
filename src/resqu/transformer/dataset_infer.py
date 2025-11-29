# dataset_infer.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

TYPECODES = [
    'B789', 'A359', 'B788', 'A332', 'A21N', 'A20N', 'A320', 'A333',
    'B738', 'A321', 'B739', 'B77W', 'B38M', 'B737', 'B772', 'B744',
    'B763', 'A319', 'B752', 'MD11', 'B77L', 'A306', 'B39M', 'A318',
    'A388', 'B748'
]

PHASE_VALUES = ["CRUISE", "DESCENT", "LEVEL", "NA"]


class FlightSequenceDatasetInference(Dataset):
    """
    Same preprocessing as training, but NO labels required.
    Returns:
      - seq_features: [L, D]
      - end_features: [D_end]
      - typecode_id:  scalar
      - idx_label:    scalar (for joining in outputs)
    """
    def __init__(
        self,
        parquet_paths,
        fixed_features,
        to_test_features,
        to_feed_end,
        missing_to_minus1=("ff_kgs_est", "mass_est", "airdist"),
        stats=None,
        max_seq_len=3600,
        step=10,  # downsample like you did in training (__getitem__ used ::10)
    ):
        self.parquet_paths = parquet_paths
        self.fixed_features = fixed_features
        self.to_test_features = to_test_features
        self.to_feed_end = to_feed_end
        self.missing_to_minus1 = set(missing_to_minus1)
        self.stats = stats
        self.max_seq_len = max_seq_len
        self.step = step

        self.numeric_features = self.fixed_features + self.to_test_features

        self.typecode2idx = {tc: i for i, tc in enumerate(TYPECODES)}
        self.phase2idx = {ph: i for i, ph in enumerate(PHASE_VALUES)}
        self.num_phases = len(self.phase2idx)

        # Build seq index across files
        self.seq_index = []
        for file_id, path in enumerate(parquet_paths):
            meta_cols = ["idx", "typecode", "phase"]
            df_meta = pd.read_parquet(path, columns=meta_cols)
            for seq_id, g in df_meta.groupby("idx"):
                tc = g["typecode"].iloc[0]
                self.seq_index.append(
                    {"file_id": file_id, "idx": seq_id, "typecode": tc}
                )

    def __len__(self):
        return len(self.seq_index)

    # ---- helpers ----

    def _normalize_numeric(self, num_arr):
        if self.stats is None:
            return np.nan_to_num(num_arr, nan=0.0)

        mean = self.stats.get("mean", {})
        std = self.stats.get("std", {})

        for j, col in enumerate(self.numeric_features):
            m = mean.get(col, 0.0)
            s = std.get(col, 1.0)
            if s == 0:
                s = 1.0

            col_vals = num_arr[:, j]
            nan_mask = np.isnan(col_vals)
            if nan_mask.any():
                col_vals[nan_mask] = m
            num_arr[:, j] = (col_vals - m) / s

        return num_arr

    def _phase_one_hot(self, phases):
        L = len(phases)
        one_hot = np.zeros((L, self.num_phases), dtype=np.float32)
        for i, ph in enumerate(phases):
            if ph in self.phase2idx:
                one_hot[i, self.phase2idx[ph]] = 1.0
        return one_hot

    # ---- core ----

    def __getitem__(self, i):
        meta = self.seq_index[i]
        file_id = meta["file_id"]
        seq_id = meta["idx"]
        typecode = meta["typecode"]
        path = self.parquet_paths[file_id]

        df_seq = pd.read_parquet(path, filters=[("idx", "=", seq_id)]).copy()
        if df_seq.empty:
            raise RuntimeError(f"Empty sequence for idx={seq_id} in file {path}")

        if "ts" in df_seq.columns:
            df_seq = df_seq.sort_values("ts").iloc[::self.step].reset_index(drop=True)

        # fill chosen cols with -1
        for col in self.missing_to_minus1:
            if col in df_seq.columns:
                df_seq[col] = df_seq[col].fillna(-1)

        # numeric timestep features
        missing_num = [c for c in self.numeric_features if c not in df_seq.columns]
        if missing_num:
            raise ValueError(f"Missing numeric features {missing_num} in {path}")

        num_arr = df_seq[self.numeric_features].to_numpy(dtype=np.float32)
        num_arr = self._normalize_numeric(num_arr)

        # phase one-hot
        if "phase" not in df_seq.columns:
            raise ValueError(f"`phase` column missing in {path}")
        phase_oh = self._phase_one_hot(df_seq["phase"])

        seq_features = np.concatenate([num_arr, phase_oh], axis=1)  # [L, D_num + 4]

        # sequence-level features
        missing_end = [c for c in self.to_feed_end if c not in df_seq.columns]
        if missing_end:
            raise ValueError(f"Missing TO_FEED_IN_THE_END columns {missing_end} in {path}")
        end_row = df_seq[self.to_feed_end].iloc[0].fillna(0.0)
        end_features = end_row.to_numpy(dtype=np.float32)

        # typecode embedding index
        if typecode not in self.typecode2idx:
            raise ValueError(f"Unknown typecode {typecode}")
        tc_id = self.typecode2idx[typecode]

        # truncate from end if longer than max_seq_len
        L = seq_features.shape[0]
        if self.max_seq_len is not None and L > self.max_seq_len:
            seq_features = seq_features[-self.max_seq_len:]

        if np.isnan(seq_features).any() or np.isnan(end_features).any():
            raise RuntimeError(f"NaNs detected in sample idx={seq_id} from {path}")

        return {
            "seq_features": torch.from_numpy(seq_features),       # [L, D]
            "end_features": torch.from_numpy(end_features),       # [D_end]
            "typecode_id": torch.tensor(tc_id, dtype=torch.long), # scalar
            "idx_label": torch.tensor(seq_id, dtype=torch.long),  # keep to join outputs
        }


def flight_collate_fn_infer(batch):
    """
    Pads variable-length seq_features (no labels).
    Returns:
      - seq_features:   [B, L_max, D]
      - end_features:   [B, D_end]
      - typecode_id:    [B]
      - idx_label:      [B]
      - attention_mask: [B, L_max] (True where valid)
    """
    B = len(batch)
    lengths = [b["seq_features"].shape[0] for b in batch]
    L_max = max(lengths)
    D = batch[0]["seq_features"].shape[1]

    seq_padded = torch.zeros(B, L_max, D, dtype=batch[0]["seq_features"].dtype)
    attn_mask = torch.zeros(B, L_max, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["seq_features"].shape[0]
        seq_padded[i, :L] = b["seq_features"]
        attn_mask[i, :L] = True

    end_features = torch.stack([b["end_features"] for b in batch], dim=0)
    typecode_ids = torch.stack([b["typecode_id"] for b in batch], dim=0)
    idx_labels = torch.stack([b["idx_label"] for b in batch], dim=0)

    return {
        "seq_features": seq_padded,
        "end_features": end_features,
        "typecode_id": typecode_ids,
        "idx_label": idx_labels,
        "attention_mask": attn_mask,
    }
