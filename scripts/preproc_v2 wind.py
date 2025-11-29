from resqu import FlightSet, data
from resqu.preprocessor import Preprocessor, FlightProc
import resqu.make_dataset as md
from fastmeteo.source import ArcoEra5
import click


@click.command
@click.option("-j", "--jobs", default=8, help="number of jobs")
@click.option(
    "-s", "--set", prompt="(train, rank, final)", help="train, rank, or final dataset"
)
@click.option("-v", "--version", default="v2_wind", help="version postfix")
def main(jobs, set, version):
    flight_set = FlightSet[set.upper()]
    if flight_set == FlightSet.TRAIN or flight_set == FlightSet.RANK:
        meteo_grid = ArcoEra5(local_store="/mnt/sda1/meteo/era5-zarr")
    elif flight_set == FlightSet.FINAL:
        meteo_grid == ArcoEra5(local_store="/mnt/sdb1/meteo/era5-zarr")

    proc = Preprocessor(
        jobs=jobs, flight_set=flight_set, version=version
    ).set_meteo_grid(meteo_grid)
    proc.run_pipe(pipe_v2_wind, meteo_grid=proc.meteo_grid)


def pipe_v2_wind(
    fid,
    flight_set,
    preproc_dir,
    meteo_grid,
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
        .get_wind(meteo_grid)
        .tas_from_wind()
        .est_mass_timeflown()
        .estimate_mass_24_chgpt()
        .est_ff()
        .est_ff_mass_tf()
        .energy()
        .work_done()
        .drop_unused_cols()
        .to_parquet(preproc_dir / f"{fid}.parquet")
    )
    return True


if __name__ == "__main__":
    # main(1, 'final', 'v2')
    main()
