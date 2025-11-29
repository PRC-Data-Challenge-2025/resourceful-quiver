import os
from pathlib import Path
import pandas as pd
from enum import Enum
import warnings

parent_dir = Path(os.path.abspath(__file__)).resolve().parent.parent.parent
data_dir = parent_dir / "data"
lgbm_data_dir = data_dir / "lgbm_dataset"
model_dir = parent_dir / "models"
train_traj_dir = data_dir / "flights_train"
rank_traj_dir = data_dir / "flights_rank"
final_traj_dir = data_dir / "flights_final"
log_dir = parent_dir / "logs"
checkpoint_dir = parent_dir / "checkpoints"
metadata_dir = data_dir / 'metadata'

if (data_dir / "fuel_train_updated.parquet").exists():
    fuel_train_data = pd.read_parquet(data_dir / "fuel_train_updated.parquet")
else:
    fuel_train_data = pd.read_parquet(data_dir / "fuel_train.parquet")
    fuel_train_data = fuel_train_data.assign(
        seg_duration=(fuel_train_data.end - fuel_train_data.start).dt.total_seconds()
    )
    fuel_train_data = fuel_train_data.assign(
        ff_kgs=fuel_train_data.fuel_kg / fuel_train_data.seg_duration
    )
    fuel_train_data.to_parquet(data_dir / "fuel_train_updated.parquet")

if (data_dir / "fuel_rank_updated.parquet").exists():
    fuel_rank_data = pd.read_parquet(data_dir / "fuel_rank_updated.parquet")
else:
    fuel_rank_data = pd.read_parquet(data_dir / "fuel_rank_submission.parquet")
    fuel_rank_data = fuel_rank_data.assign(
        seg_duration=(fuel_rank_data.end - fuel_rank_data.start).dt.total_seconds(),
        ff_kgs=None,
    )
    fuel_rank_data.to_parquet(data_dir / "fuel_rank_updated.parquet")

if (data_dir / "fuel_final_updated.parquet").exists():
    fuel_final_data = pd.read_parquet(data_dir / "fuel_final_updated.parquet")
else:
    fuel_final_data = pd.read_parquet(data_dir / "fuel_final_submission.parquet")
    fuel_final_data = fuel_final_data.assign(
        seg_duration=(fuel_final_data.end - fuel_final_data.start).dt.total_seconds(),
        ff_kgs=None,
    )
    fuel_final_data.to_parquet(data_dir / "fuel_final_updated.parquet")

airport_data = pd.read_parquet(data_dir / "apt_updated.parquet")
flight_list_train = pd.read_parquet(data_dir / "flightlist_train.parquet")
flight_list_rank = pd.read_parquet(data_dir / "flightlist_rank.parquet")
flight_list_final = pd.read_parquet(data_dir / "flightlist_final.parquet")

ac_tows = pd.read_csv(data_dir / "ac_tows.csv").rename(columns={"ac": "aircraft_type"})


class FlightSet(Enum):
    UNKNOWN = 0
    TRAIN = 1
    RANK = 2
    FINAL = 3

    @classmethod
    def get_set(cls, fid: str) -> Enum:
        if (train_traj_dir / f"{fid}.parquet").exists():
            return cls.TRAIN
        elif (rank_traj_dir / f"{fid}.parquet").exists():
            return cls.RANK
        elif (final_traj_dir / f"{fid}.parquet").exists():
            return cls.FINAL
        else:
            warnings.warn(f"flight {fid} not found in train and rank!!!!")
            return cls.UNKNOWN


def get_rawraw_df(fid, flight_set: FlightSet | None = None):
    if flight_set is None:
        flight_set = FlightSet.get_set(fid)

    if flight_set == FlightSet.TRAIN:
        df_path = train_traj_dir / f"{fid}.parquet"
    elif flight_set == FlightSet.RANK:
        df_path = rank_traj_dir / f"{fid}.parquet"
    elif flight_set == FlightSet.FINAL:
        df_path = final_traj_dir / f"{fid}.parquet"
    else:
        warnings.warn(f"flight {fid} not found in train and rank!!!!")
        return None
    return pd.read_parquet(df_path)


def get_raw_df(fid, flight_set: FlightSet | None = None):
    if flight_set is None:
        flight_set = FlightSet.get_set(fid)

    if flight_set == FlightSet.TRAIN:
        df_path = train_traj_dir / f"{fid}.parquet"
    elif flight_set == FlightSet.RANK:
        df_path = rank_traj_dir / f"{fid}.parquet"
    elif flight_set == FlightSet.FINAL:
        df_path = final_traj_dir / f"{fid}.parquet"
    else:
        warnings.warn(f"flight {fid} not found in train and rank!!!!")
        return None

    df = (
        pd.read_parquet(df_path)
        .drop_duplicates(subset="timestamp")
        .sort_values(by="timestamp")
        .reset_index(drop=True)
    )

    # drop duplicate in latitude and longitude that are not nan
    valid_mask = ~(
        df.latitude.notna()
        & df.longitude.notna()
        & df.duplicated(subset=["latitude", "longitude"])
    )
    # drop latitude that are not within (-90, 90) degrees
    valid_mask &= (df.latitude <= 90) & (df.latitude >= -90)
    # drop longitude that are not with (-180, 180) degrees
    valid_mask &= (df.longitude >= -180) & (df.longitude <= 180)

    # drop data where ground speed that are too slow while above certain altitude, could be because of gps jamming
    # valid_mask &= ~((df.groundspeed <= 200) & (df.altitude >= 20000))
    df = df[valid_mask].reset_index(drop=True)
    return df


def get_fuel_set(flight_set: FlightSet):
    if flight_set == FlightSet.TRAIN:
        return fuel_train_data
    elif flight_set == FlightSet.RANK:
        return fuel_rank_data
    elif flight_set == FlightSet.FINAL:
        return fuel_final_data
    else:
        return None


def get_flist_set(flight_set: FlightSet):
    if flight_set == FlightSet.TRAIN:
        return flight_list_train
    elif flight_set == FlightSet.RANK:
        return flight_list_rank
    elif flight_set == FlightSet.FINAL:
        return flight_list_final
    else:
        return None


def get_flist_id(fid, flight_set: FlightSet | None = None):
    if flight_set is None:
        flight_set = FlightSet.get_set(fid)

    if flight_set == FlightSet.TRAIN:
        f_info = flight_list_train[flight_list_train.flight_id == fid]
    elif flight_set == FlightSet.RANK:
        f_info = flight_list_rank[flight_list_rank.flight_id == fid]
    elif flight_set == FlightSet.FINAL:
        f_info = flight_list_final[flight_list_final.flight_id == fid]
    else:
        warnings.warn(f"flight {fid} not found in train, rank, and final!!!!")
        return None
    return f_info.squeeze()


def get_fuel_id(fid, flight_set: FlightSet | None = None):
    if flight_set is None:
        flight_set = FlightSet.get_set(fid)

    if flight_set == FlightSet.TRAIN:
        fuel_data = fuel_train_data[fuel_train_data.flight_id == fid]
    elif flight_set == FlightSet.RANK:
        fuel_data = fuel_rank_data[fuel_rank_data.flight_id == fid]
    elif flight_set == FlightSet.FINAL:
        fuel_data = fuel_final_data[fuel_final_data.flight_id == fid]
    else:
        warnings.warn(f"flight {fid} not found in train, rank, and final!!!!")
        return None
    return fuel_data
