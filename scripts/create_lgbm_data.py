from resqu import FlightSet, data
import resqu.make_dataset as md

import click


@click.command
@click.option("-j", "--jobs", default=8, help="number of jobs")
@click.option("-v", "--version", default="v0", help="version postfix")
def main(jobs, version):
    ver = version
    in_dir_train = data.data_dir / f"flights_train_preproc_{ver}"
    in_dir_rank = data.data_dir / f"flights_rank_preproc_{ver}"
    in_dir_final = data.data_dir / f"flights_final_preproc_{ver}"

    agg_funcs = ["mean", "std", "min", "max"]
    split_funcs = ["mean", "std"]
    split_feat = ["vertical_rate"]
    # split_feat = 'all'
    split = 9

    # split_feat = None
    # split = 0

    df_train = md.make_lgbm_dataset(
        in_dir=in_dir_train,
        flight_set=data.FlightSet.TRAIN,
        agg_funcs=agg_funcs,
        split_funcs=split_funcs,
        split_feat=split_feat,
        split=split,
    )
    df_rank = md.make_lgbm_dataset(
        in_dir=in_dir_rank,
        flight_set=data.FlightSet.RANK,
        agg_funcs=agg_funcs,
        split_funcs=split_funcs,
        split_feat=split_feat,
        split=split,
    )

    df_final = md.make_lgbm_dataset(
        in_dir=in_dir_final,
        flight_set=data.FlightSet.FINAL,
        agg_funcs=agg_funcs,
        split_funcs=split_funcs,
        split_feat=split_feat,
        split=split,
    )
    
    df_train.to_parquet(data.lgbm_data_dir / f"df_train_{ver}.parquet")
    df_rank.to_parquet(data.lgbm_data_dir / f"df_rank_{ver}.parquet")
    df_final.to_parquet(data.lgbm_data_dir / f"df_final_{ver}.parquet")


if __name__ == "__main__":
    main()
