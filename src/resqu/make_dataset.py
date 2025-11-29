import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from . import data
from .data import FlightSet
from .preprocessor import FlightProc
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
from typing import List


def process_flight_raw(
    fid: str,
    in_dir: Path,
    out_dir: Path,
    flight_set: FlightSet | None = None,
):
    """
    Args:
        fid: flight id
        in_dir: where preprocessed flights are stored
        out_dir: where postprocessed dfs will be stored
        fuel_df: fuel data corresponding to the flight id
        flight_set: FlightSet.TRAIN|RANK|FINAL

    """
    out_dir.mkdir(exist_ok=True)

    df = pd.read_parquet(in_dir / f"{fid}.parquet")
    fuel = data.get_fuel_id(fid, flight_set)

    df = df.assign(idx=None)
    for _, row in fuel.iterrows():
        start = row.start
        end = row.end

        seg_mask = (df.timestamp >= start) & (df.timestamp <= end)
        df.loc[seg_mask, 'idx'] = row.idx
    df[df.idx.notna()].reset_index(drop=True).to_parquet(out_dir / f"{fid}.parquet")
    return

def process_flight_agg(
    fid: str,
    in_dir: Path,
    flight_set: FlightSet | None = None,
    agg_funcs: List = ["mean", "std"],
):
    """
    Args:
        fid: flight_id
        in_dir: where preprocessed flights are stored
        flight_set: FlightSet.Train|RANK|FINAL
        agg_funcs: function to be passed to DataFrame.GroupBy.agg
    """

    df = pd.read_parquet(in_dir / f"{fid}.parquet")
    f_info = data.get_flist_id(
        fid, flight_set
    )  # returns a series, because it's only one row
    info_col = f_info.index.drop(
        ["origin_name", "destination_name", "origin_icao", "destination_icao"]
    )
    fuel = data.get_fuel_id(fid, flight_set)
    fuel = fuel.merge(
        f_info[info_col].to_frame().transpose(), how="left", on="flight_id"
    )
    t_to = f_info.takeoff
    t_ld = f_info.landed
    flight_dur = (t_ld - t_to).total_seconds()

    # assign fixed features
    fuel = fuel.assign(
        flight_duration=flight_dur,
        tow_est_kg=df.tow_est_kg.iloc[0],
        tau_s=(fuel.start - t_to).dt.total_seconds() / flight_dur,
        tau_e=(fuel.end - t_to).dt.total_seconds() / flight_dur,
        full_flight_dist=df.distance_km.max(),
        seg_dist=np.nan,
    )

    # assign fuel idx to flight segments
    df = df.assign(idx=None)

    for i, row in fuel.iterrows():
        # assign segments to fuel idx
        mask = (df.timestamp >= row.start) & (df.timestamp <= row.end)
        df.loc[mask, "idx"] = row.idx
        fuel.loc[i, "seg_dist"] = (
            df[mask].distance_km.max() - df[mask].distance_km.min()
        )

    agg_feat = df.columns[df.dtypes == "float64"].drop(labels=["tow_est_kg"])

    cat_feat = ["phase"]

    df_by_idx = df.groupby(["idx"])

    # merge categorical features based on the first value
    df_cat = df_by_idx[cat_feat].first()
    fuel = fuel.merge(df_cat, how="left", on="idx")

    if len(agg_feat) > 0:
        # aggregate data no split
        df_agg = df_by_idx[agg_feat].agg(agg_funcs)
        df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
        fuel = fuel.merge(df_agg.reset_index(), how="left", on="idx")

    return fuel


def process_flight_split(
    fid: str,
    in_dir: Path,
    flight_set: FlightSet | None = None,
    split_funcs: List = ["mean", "std"],
    split_feat: List | pd.Index | str | None = None,
    split: int = 0,
):
    """
    Args:
        fid: flight_id
        in_dir: where preprocessed flights are stored
        flight_set: FlightSet.Train|RANK|FINAL
        agg_funcs: functions to be passed to DataFrame.GroupBy.agg
        split_feat: features columns you want to split, must be list or Index, if it is str, must be 'all'
        split: number of even split based on segment time, split=1 will divide the features in to 2 parts
    """
    if split_feat is not None and (split is None or split < 1):
        print("split feature provided but split value is invalid")
        return None

    df = pd.read_parquet(in_dir / f"{fid}.parquet")

    fuel = data.get_fuel_id(fid, flight_set)

    # assign fuel idx and split to flight segments
    df = df.assign(idx=None, split=None)

    for i, row in fuel.iterrows():
        # assign segments to fuel idx
        mask = (df.timestamp >= row.start) & (df.timestamp <= row.end)
        df.loc[mask, "idx"] = row.idx

        # split the segments based on time
        if split:
            shift = pd.to_timedelta(row.seg_duration / (split + 1), "s")
            for i in range(split + 1):
                split_mask = (df.timestamp >= (row.start + i * shift)) & (
                    df.timestamp <= row.start + (i + 1) * shift
                )
                df.loc[split_mask, "split"] = i

    if type(split_feat) is str and split_feat == "all":
        split_feat = df.columns[df.dtypes == "float64"].drop(labels=["tow_est_kg"])

    if split:
        # aggregate data by split
        df_by_split = df.groupby(["idx", "split"])
        df_agg_split = df_by_split[split_feat].agg(split_funcs).unstack(level="split")
        df_agg_split.columns = [
            f"{col[0]}_{col[1]}_{col[2]}" for col in df_agg_split.columns
        ]

        return df_agg_split.reset_index()


def infer_from_dir(dir: Path):
    name = dir.name.lower()
    if "train" in name:
        flight_set = FlightSet.TRAIN
    elif "rank" in name:
        flight_set = FlightSet.RANK
    elif "final" in name:
        flight_set = FlightSet.FINAL
    else:
        print("no flight set specified and cannot be inferred from the directory")
        return None

    return flight_set


def run_proc(
    process_func,
    in_dir: Path,
    flight_set: FlightSet | None = None,
    max_workers: int | None = None,
    **kwargs,
):
    """
    Use processpool executer to run your process_func that makes a dataset
    by default in_dir will be mapped to process_func using functools.partial
    additional static argument of process_func can be provided as kwargs and will be mapped
    Args:
        process_func: processing function to be submitted to the processpool, by default it should have the signature
                        ( in_dir, fid, fuel_df)
                        fid: flight_id
                        in_dir: directory of the processed flights
        in_dir: input directory to the process_func, where the preprocessed flights sit
        flight_set: TRAIN | RANK | FINAL
        max_workers: Use all cores by default, or specify an integer
        **kwargs: static inputs to be mapped to process_func
    Return:
        all_results: list of results from process_func
    """
    if not in_dir.exists():
        print("no input directory found")
        return None
    if flight_set is None:
        # infer flight_set from in_dir
        flight_set = infer_from_dir(in_dir)

    fids = [f.stem for f in in_dir.glob("*.parquet")]

    func = functools.partial(
        process_func, in_dir=in_dir, flight_set=flight_set, **kwargs
    )
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(func, fid=fid): fid for fid in fids}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing flights"
        ):
            try:
                all_results.append(future.result())
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    return all_results


def make_lgbm_dataset(
    in_dir: Path,
    out_dir: Path | None = None,
    flight_set: FlightSet | None = None,
    agg_funcs: List = ["mean", "std"],
    split_funcs: List = ["mean", "std"],
    split_feat: List | pd.Index | str = None,
    split: int = 0,
):
    result_agg = run_proc(
        process_func=process_flight_agg,
        in_dir=in_dir,
        flight_set=flight_set,
        agg_funcs=agg_funcs,
    )
    df = (
        pd.concat(result_agg)
        .sort_values(by="idx")
        .reset_index(drop=True)
        .merge(
            data.ac_tows.drop(labels=["min_tow_ch_set", "max_tow_ch_set"], axis=1),
            on="aircraft_type",
            how="left",
        )
    )
    if split_feat is not None and split > 0:
        result_split = run_proc(
            process_func=process_flight_split,
            in_dir=in_dir,
            flight_set=flight_set,
            split_funcs=split_funcs,
            split_feat=split_feat,
            split=split,
        )
        df = df.merge(
            pd.concat(result_split).reset_index(drop=True), on="idx", how="left"
        )
    if out_dir is not None:
        out_dir.mkdir(exist_ok=True)
        file_name = f"df_lgbm_{flight_set.name.lower()}.parquet"
        df.to_parquet(out_dir / file_name)
    return df


def make_raw_dataset(in_dir: Path, out_dir: Path, flight_set: FlightSet | None = None):
    run_proc(
        process_func=process_flight_raw,
        in_dir=in_dir,
        out_dir=out_dir,
    )
