import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_global_stats(data_dir: str):
    per_file_stats = []

    # Step 1: Collect per-file statistics
    for fname in tqdm(os.listdir(data_dir), desc="Processing files"):
        path = os.path.join(data_dir, fname)
        if not path.endswith(".parquet"):
            continue

        df = pd.read_parquet(path)
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            continue

        # Drop columns that are all NaN
        num = num.dropna(axis=1, how="all")
        if num.empty:
            continue

        n = num.count()  # count non-NaN values per column
        means = num.mean(axis=0, skipna=True)
        stds = num.std(axis=0, ddof=0, skipna=True)  # population std

        per_file_stats.append({"n": n, "mean": means, "std": stds})

    if not per_file_stats:
        raise ValueError("No numeric data found in directory.")

    # Step 2: Merge statistics correctly
    global_n = None
    global_mean = None
    global_M2 = None  # sum of squared deviations

    for stats in per_file_stats:
        n_i = stats["n"]
        mean_i = stats["mean"]
        var_i = stats["std"] ** 2

        # Align columns dynamically
        cols = mean_i.index
        if global_mean is None:
            global_n = n_i.astype(float)
            global_mean = mean_i.astype(float)
            global_M2 = var_i * n_i
        else:
            # Align to handle missing columns between files
            all_cols = global_mean.index.union(cols)
            global_mean = global_mean.reindex(all_cols, fill_value=np.nan)
            global_M2 = global_M2.reindex(all_cols, fill_value=0.0)
            global_n = global_n.reindex(all_cols, fill_value=0.0)

            mean_i = mean_i.reindex(all_cols)
            var_i = var_i.reindex(all_cols)
            n_i = n_i.reindex(all_cols, fill_value=0.0)

            delta = mean_i - global_mean
            total_n = global_n + n_i
            valid = (n_i > 0) & (global_n > 0)

            # Update only where valid data exists
            global_M2[valid] = (
                global_M2[valid]
                + var_i[valid] * n_i[valid]
                + (delta[valid] ** 2) * (global_n[valid] * n_i[valid] / total_n[valid])
            )

            global_mean[valid] = global_mean[valid] + delta[valid] * (n_i[valid] / total_n[valid])
            global_n = total_n

    # Final global stats
    global_var = global_M2 / global_n
    global_std = np.sqrt(global_var)

    # Drop columns that never had any valid value
    valid_cols = global_n[global_n > 0].index
    global_mean = global_mean[valid_cols]
    global_std = global_std[valid_cols]
    global_n = global_n[valid_cols]

    return {
        "mean": global_mean.to_dict(),
        "std": global_std.to_dict(),
        "n_total_per_column": global_n.astype(int).to_dict(),
    }


if __name__ == "__main__":
    DATA_DIR = "train_data_raw"  # change this if needed
    global_stats = compute_global_stats(DATA_DIR)
    print(global_stats)
