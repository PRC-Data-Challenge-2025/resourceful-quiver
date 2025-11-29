from resqu import FlightSet, data
from resqu.preprocessor import Preprocessor, FlightProc
from fastmeteo.source import ArcoEra5

import click


@click.command
@click.option("-j", "--jobs", default=8, help="number of jobs")
@click.option(
    "-s", "--set", prompt="(train, rank, final)", help="train, rank, or final dataset"
)
def main(jobs, set):
    flight_set = FlightSet[set.upper()]
    if flight_set == FlightSet.TRAIN or flight_set == FlightSet.RANK:
        meteo_grid = ArcoEra5(local_store="/mnt/sda1/meteo/era5-zarr")
    elif flight_set == FlightSet.FINAL:
        meteo_grid ==  ArcoEra5(local_store="/mnt/sdb1/meteo/era5-zarr")
    
    proc = Preprocessor(
        jobs=jobs, flight_set=flight_set, version="v0_wind"
    ).set_meteo_grid(meteo_grid)
    proc.run_pipe(pipe_wind, meteo_grid = proc.meteo_grid)


def pipe_wind(fid, flight_set, preproc_dir, meteo_grid):
    df = data.get_raw_df(fid, flight_set)
    flight = FlightProc(df, flight_set)
    (
        flight.filter()
        .resample("1s", projection="lcc")
        .ts()
        .phases()
        .est_dist()
        .tas_from_mach_cas()
        .get_wind(meteo_grid)
        .tas_from_wind()
        .estimate_mass_24_chgpt()
        .est_ff()
        .to_parquet(preproc_dir / f"{fid}.parquet")
    )
    return True


if __name__ == "__main__":
    main()
