import click
from . import data
from .data import FlightSet


@click.command()
@click.option("-j", "--jobs", default=8, help="number of jobs")
@click.option(
    "-s", "--set", prompt="(train, rank, final)", help="train, rank, or final dataset"
)
@click.option("-v", "--version", help="preprocess version")
@click.option("--skip", default=False, help="skip already processed flights")
@click.option("-m", "--method", default='v0', help="presets are [v0, v1, v2]")
def preproc_main(jobs, set, version, skip, method):
    from .preprocessor import Preprocessor
    from fastmeteo.source import ArcoEra5

    if version is not None:
        try:
            version = int(version)
        except ValueError:
            version = str(version)

    if set == "train":
        meteo_grid = ArcoEra5(local_store="/mnt/sda1/meteo/era5-zarr")
        flight_list = data.flight_list_train.sort_values(by="takeoff")

        processor = (
            Preprocessor(
                jobs=jobs,
                flight_set=FlightSet.TRAIN,
                version=version,
            )
            .set_meteo_grid(meteo_grid)
        )

    elif set == "rank":
        meteo_grid = ArcoEra5(local_store="/mnt/sda1/meteo/era5-zarr")
        # meteo_grid = '/mnt/sda1/meteo/era5-zarr'
        processor = (
            Preprocessor(
                jobs=jobs,
                flight_set=FlightSet.RANK,
                version=version,
            )
            .set_meteo_grid(meteo_grid)
        )

    elif set == "final":
        meteo_grid = ArcoEra5(local_store="/mnt/sdb1/meteo/era5-zarr")
        processor = (
            Preprocessor(
                jobs=jobs,
                flight_set=FlightSet.FINAL,
                version=version,
            )
            .set_meteo_grid(meteo_grid)
        )


    if method == 'v0':
        processor.run_pipe(pipe=Preprocessor.pipe_v0)
    elif method == 'v1':
        processor.run_pipe(pipe=Preprocessor.pipe_v1)
    elif method == 'v2':
        processor.run_pipe(pipe=Preprocessor.pipe_v2, meteo_grid=processor.meteo_grid)

    # not implemented yet, do we even need it?
    if skip:
        fids = [f.stem for f in processor.preproc_dir.glob("*.parquet")]
        proc_mask = ~flight_list.flight_id.isin(fids)
        flight_list = flight_list[proc_mask]

@click.command()
@click.option('--raw', default=False, is_flag=True)
@click.option('-i', '--in-dir', prompt='specify directory of preprocessed flights', type=str)
@click.option('-o', '--out-dir', prompt='specify directory of the output dataset', type=str)
@click.option('-s', '--set', 'flight_set', type=str)
def make_dataset_main(raw, in_dir, out_dir, flight_set):
    from . import make_dataset
    from pathlib import Path
    in_dir = Path(in_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if flight_set == 'train':
        flight_set = FlightSet.TRAIN
    elif flight_set == 'rank':
        flight_set = FlightSet.RANK
    elif flight_set == 'final':
        flight_set = FlightSet.FINAL
    else:
        flight_set = None

    if raw:
        make_dataset.make_raw_dataset(in_dir, out_dir, flight_set)
    
    else:
        make_dataset.make_lgbm_dataset(in_dir, out_dir, flight_set)