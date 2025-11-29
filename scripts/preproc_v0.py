from resqu import FlightSet, data
from resqu.preprocessor import Preprocessor, FlightProc
import resqu.make_dataset as md

import click


@click.command
@click.option("-j", "--jobs", default=8, help="number of jobs")
@click.option(
    "-s", "--set", prompt="(train, rank, final)", help="train, rank, or final dataset"
)
@click.option("-v", "--version", default="v0", help="version postfix")
def main(jobs, set, version):
    flight_set = FlightSet[set.upper()]
    proc = Preprocessor(jobs=jobs, flight_set=flight_set, version=version)
    proc.run_pipe(Preprocessor.pipe_v0)


def pipe_v0_1(
    fid,
    flight_set,
    preproc_dir,
):
    df = data.get_rawraw_df(fid, flight_set)
    flight = FlightProc(df, flight_set)
    (
        flight.filter()
        .resample("1s", projection="lcc")
        .phases()
        .ts()
        .est_dist()
        .tas_from_mach_cas()
        .estimate_mass_24_chgpt()
        .est_ff()
        .drop_unused_cols()
        .to_parquet(preproc_dir / f"{fid}.parquet")
    )
    return True


def pipe_v0_2(
    fid,
    flight_set,
    preproc_dir,
):
    df = data.get_rawraw_df(fid, flight_set)
    flight = FlightProc(df, flight_set)
    (
        flight.filter()
        .resample("1s", projection="lcc")
        .phases()
        .ts()
        .est_dist()
        .tas_from_mach_cas()
        .est_mass_timeflown()
        .est_ff()
        .drop_unused_cols()
        .to_parquet(preproc_dir / f"{fid}.parquet")
    )
    return True

if __name__ == "__main__":
    main()
