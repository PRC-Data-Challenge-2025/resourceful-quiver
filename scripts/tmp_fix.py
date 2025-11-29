"""
Quick and dirty postprocessing fix, write your custom pipeline to manipulate data
"""

from resqu import FlightSet, data
from resqu.preprocessor import Preprocessor, FlightProc
import resqu.make_dataset as md
import pandas as pd
import click


@click.command
@click.option("-j", "--jobs", default=8, help="number of jobs")
@click.option(
    "-s", "--set", prompt="(train, rank, final)", help="train, rank, or final dataset"
)
def main(jobs, set):
    flight_set = FlightSet[set.upper()]
    proc = Preprocessor(jobs=jobs, flight_set=flight_set, version="v0_wind")
    if flight_set == FlightSet.TRAIN:
        in_dir = data.data_dir / "flights_train_preproc_v0_wind_"
    elif flight_set == FlightSet.RANK:
        in_dir = data.data_dir / "flights_rank_preproc_v0_wind_"
    proc.run_pipe(pipe_fix, in_dir=in_dir)


def pipe_fix(
    fid,
    flight_set,
    in_dir,
    preproc_dir,
):
    df = pd.read_parquet(in_dir / f"{fid}.parquet")
    flight = FlightProc(df, flight_set)
    # (flight.ts().est_dist().drop_unused_cols().to_parquet(preproc_dir / f"{fid}.parquet"))
    flight.mach_CAS_from_compute_TAS().to_parquet(preproc_dir / f"{fid}.parquet")
    return True


if __name__ == "__main__":
    main()
