import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


TYPECODES = [
    'B789', 'A359', 'B788', 'A332', 'A21N', 'A20N', 'A320', 'A333',
    'B738', 'A321', 'B739', 'B77W', 'B38M', 'B737', 'B772', 'B744',
    'B763', 'A319', 'B752', 'MD11', 'B77L', 'A306', 'B39M', 'A318',
    'A388', 'B748'
]

PHASE_VALUES = ["CRUISE", "DESCENT", "LEVEL", "NA"]


class FlightSequenceDataset(Dataset):
    def __init__(
        self,
        parquet_paths,
        fixed_features,
        to_test_features,
        to_feed_end,
        missing_to_minus1=("ff_kgs_est", "mass_est", "airdist"),
        stats={'mean': {'CAS': 140.01529166159094, 'TAS': 454.9053924817837, 'airdist': 3397.329409108582, 'altitude': 33434.47870860869, 'compute_gs': 438.1911930008268, 'compute_track': 186.2023623763514, 'cumdist': 1767.1369651131813, 'deltat': 1125.0190441144714, 'distance_km': 3272.7376593896265, 'ff_kgs_est': 1.2150995758031118, 'fuel_kg': 1305.6280004880225, 'groundspeed': 454.1061016738322, 'idx': 61609.8734092951, 'latitude': 28.39102746357758, 'longitude': -32.548034576877676, 'mach': 0.7827150290878517, 'mass_est': 114451.58724705917, 'specific_humidity': 0.0006663787092651272, 'temperature': 229.2314779614455, 'tow_est_kg': 135088.48518138775, 'track': 174.6883780359047, 'track_unwrapped': 161.32431085741953, 'ts': 13745.19522822413, 'u_component_of_wind': 12.171811299484274, 'v_component_of_wind': -0.5444254829844769, 'vertical_rate': -19.433975247357875, 'x': -709720.1553305085, 'y': 317649.885759743},
                'std': {'CAS': 154.91858852257167, 'TAS': 323.64203710270533, 'airdist': 7028.8395394463, 'altitude': 9523.96634155301, 'compute_gs': 1050.805126840548, 'compute_track': 100.86986478621897, 'cumdist': 1602.0794105613377, 'deltat': 925.2511058831888, 'distance_km': 2967.0510683596085, 'ff_kgs_est': 0.8404119682908922, 'fuel_kg': 1526.9363191840866, 'groundspeed': 325.1697646087668, 'idx': 37752.574993560884, 'latitude': 18.911142335498052, 'longitude': 91.0382779082025, 'mach': 0.5607426539325037, 'mass_est': 57844.6075161093, 'specific_humidity': 0.0024349654964718245, 'temperature': 20.017739109524978, 'tow_est_kg': 74275.61286143982, 'track': 98.23456165218542, 'track_unwrapped': 144.44895680098898, 'ts': 10946.902583732199, 'u_component_of_wind': 15.67457111943422, 'v_component_of_wind': 11.497200504313675, 'vertical_rate': 629.0541655173895, 'x': 1735220.4786363742, 'y': 1515446.4601253234},
                'n_total_per_column': {'CAS': 68967615, 'TAS': 68967615, 'airdist': 68967073, 'altitude': 68967615, 'compute_gs': 68967615, 'compute_track': 68967615, 'cumdist': 68967615, 'deltat': 68967615, 'distance_km': 68967615, 'ff_kgs_est': 68936470, 'fuel_kg': 68967615, 'groundspeed': 68967615, 'idx': 68967615, 'latitude': 68967615, 'longitude': 68967615, 'mach': 68967615, 'mass_est': 68936470, 'specific_humidity': 68967615, 'temperature': 68967615, 'tow_est_kg': 68967615, 'track': 68967615, 'track_unwrapped': 68967615, 'ts': 68967615, 'u_component_of_wind': 68967615, 'v_component_of_wind': 68967615, 'vertical_rate': 68967615, 'x': 68967615, 'y': 68967615}},                 # optional: {"mean": {col:..}, "std": {col:..}}
        max_seq_len=3600,
    ):
        """
        One sequence = all rows with same `idx` in a parquet file.

        Per-timestep token:
          [FIXED_FEATURES + TO_TEST_FEATURES + one_hot(phase: 4)]

        Sequence-level (fed with CLS in the head):
          - typecode_id  (26-way from fixed list)
          - TO_FEED_IN_THE_END (e.g. tow_est_kg, deltat)

        Labels:
          - idx_label
          - fuel_kg_label = sum over sequence
        """
        self.parquet_paths = parquet_paths
        self.fixed_features = fixed_features
        self.to_test_features = to_test_features
        self.to_feed_end = to_feed_end
        self.missing_to_minus1 = set(missing_to_minus1)
        self.stats = stats
        self.max_seq_len = max_seq_len

        # timestep-level numeric
        self.numeric_features = self.fixed_features + self.to_test_features

        # fixed mappings
        self.typecode2idx = {tc: i for i, tc in enumerate(TYPECODES)}
        self.num_typecodes = len(self.typecode2idx)

        self.phase2idx = {ph: i for i, ph in enumerate(PHASE_VALUES)}
        self.num_phases = len(self.phase2idx)

        # build seq index: [{file_id, idx, typecode}, ...]
        self.seq_index = []
        for file_id, path in tqdm(enumerate(parquet_paths)):
            meta_cols = ["idx", "typecode", "phase"]
            df_meta = pd.read_parquet(path, columns=meta_cols)

            for seq_id, g in df_meta.groupby("idx"):
                tc = g["typecode"].iloc[0]
                # unknown typecodes mapped to -1, you can assert instead:
                # if tc not in self.typecode2idx: continue / raise
                self.seq_index.append(
                    {
                        "file_id": file_id,
                        "idx": seq_id,
                        "typecode": tc,
                    }
                )
        # for file_id, path in enumerate(parquet_paths.glob('*.parquet')):
        #     meta_cols = ["idx", "typecode", "phase"]
        #     df_meta = pd.read_parquet(path, columns=meta_cols)

        #     for seq_id, g in df_meta.groupby("idx"):
        #         tc = g["typecode"].iloc[0]
        #         # unknown typecodes mapped to -1, you can assert instead:
        #         # if tc not in self.typecode2idx: continue / raise
        #         self.seq_index.append(
        #             {
        #                 "file_id": file_id,
        #                 "idx": seq_id,
        #                 "typecode": tc,
        #             }
        #         )

    def __len__(self):
        return len(self.seq_index)

    # ---- helpers ----

    def _normalize_numeric(self, num_arr):
        if self.stats is None:
            # Simple safe default: replace NaNs with 0
            return np.nan_to_num(num_arr, nan=0.0)

        mean = self.stats.get("mean", {})
        std = self.stats.get("std", {})

        for j, col in enumerate(self.numeric_features):
            m = mean.get(col, 0.0)
            s = std.get(col, 1.0)
            if s == 0:
                s = 1.0

            col_vals = num_arr[:, j]

            # fill NaNs with mean before normalization
            nan_mask = np.isnan(col_vals)
            if nan_mask.any():
                col_vals[nan_mask] = m

            num_arr[:, j] = (col_vals - m) / s

        return num_arr


    def _phase_one_hot(self, phases):
        """
        phases: Series of phase strings.
        -> [L, num_phases] one-hot, unknown/NaN -> all zeros.
        """
        L = len(phases)
        one_hot = np.zeros((L, self.num_phases), dtype=np.float32)
        for i, ph in enumerate(phases):
            if ph in self.phase2idx:
                j = self.phase2idx[ph]
                one_hot[i, j] = 1.0
        return one_hot

    # ---- core ----

    def __getitem__(self, i):
        meta = self.seq_index[i]
        file_id = meta["file_id"]
        seq_id = meta["idx"]
        typecode = meta["typecode"]
        path = self.parquet_paths[file_id]

        df_seq = pd.read_parquet(
            path,
            filters=[("idx", "=", seq_id)],
        ).copy()

        if df_seq.empty:
            raise RuntimeError(f"Empty sequence for idx={seq_id} in file {path}")

        if "ts" in df_seq.columns:
            df_seq = df_seq.sort_values("ts")
            df_seq = df_seq.iloc[::10].reset_index(drop=True)
            # if len(df_seq)>3000:
            #     df_seq = df_seq.iloc[::10].reset_index(drop=True)
            # elif len(df_seq)>2000 and len(df_seq)<3000:
            #     df_seq = df_seq.iloc[::8].reset_index(drop=True)
            # elif len(df_seq)>1500 and len(df_seq)<2000:
            #     df_seq = df_seq.iloc[::6].reset_index(drop=True)
            # elif len(df_seq)>1000 and len(df_seq)<1500:
            #     df_seq = df_seq.iloc[:5].reset_index(drop=True)
            # elif len(df_seq)>500 and len(df_seq)<1000:
            #     df_seq = df_seq.iloc[::3].reset_index(drop=True)
            # elif len(df_seq)>200 and len(df_seq)<500:
            #     df_seq = df_seq.iloc[::2].reset_index(drop=True)
            # else:
            #     pass
            

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

        # seq-level features (tow_est_kg, deltat, etc.)
        missing_end = [c for c in self.to_feed_end if c not in df_seq.columns]
        if missing_end:
            raise ValueError(f"Missing TO_FEED_IN_THE_END columns {missing_end} in {path}")

        end_row = df_seq[self.to_feed_end].iloc[0].fillna(0.0)
        end_features = end_row.to_numpy(dtype=np.float32)


        # typecode id (if unknown, set -1 so model can handle / assert here)
        if typecode not in self.typecode2idx:
            raise ValueError(f"Unknown typecode {typecode}")
        tc_id = self.typecode2idx[typecode]


        # label: fuel_kg sum
        if "fuel_kg" not in df_seq.columns:
            raise ValueError("`fuel_kg` column not found; required for label.")

        fuel_kg = df_seq["fuel_kg"].iloc[-1].astype(np.float32)


        # truncate from end if longer than max_seq_len
        L = seq_features.shape[0]
        if self.max_seq_len is not None and L > self.max_seq_len:
            seq_features = seq_features[-self.max_seq_len:]
            L = self.max_seq_len

        sample = {
            "seq_features": torch.from_numpy(seq_features),         # [L, D]
            "end_features": torch.from_numpy(end_features),         # [D_end]
            "typecode_id": torch.tensor(tc_id, dtype=torch.long),   # scalar
            "idx_label": torch.tensor(seq_id, dtype=torch.long),
            "fuel_kg_label": torch.tensor(fuel_kg, dtype=torch.float32),
        }

        if (
            np.isnan(seq_features).any()
            or np.isnan(end_features).any()
            or np.isnan(fuel_kg)
        ):
            raise RuntimeError(f"NaNs detected in sample idx={seq_id} from {path}")

        return sample


def flight_collate_fn(batch):
    """
    Pads variable-length seq_features.

    Returns:
      - seq_features:   [B, L_max, D]
      - end_features:   [B, D_end]
      - typecode_id:    [B]
      - idx_label:      [B]
      - fuel_kg_label:  [B]
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
    fuel_kg_labels = torch.stack([b["fuel_kg_label"] for b in batch], dim=0)

    return {
        "seq_features": seq_padded,
        "end_features": end_features,
        "typecode_id": typecode_ids,
        "idx_label": idx_labels,
        "fuel_kg_label": fuel_kg_labels,
        "attention_mask": attn_mask,
    }
