# %%
import pandas as pd
import numpy as np
import re
import subprocess
import optuna
import random

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor, early_stopping

# -------------------------------------------------
# Feature definition
# -------------------------------------------------
default_cols = [
    "seg_duration", "aircraft_type", "full_flight_dist",
    "altitude_mean", "track_mean", "track_std",
    "vertical_rate_mean", "phase", "seg_dist",
    "vertical_rate_std", "groundspeed_mean", "groundspeed_std",
    "mach_mean", "mach_std", "TAS_mean", "TAS_std",
    "CAS_mean", "CAS_std",
    "vertical_rate_min", "vertical_rate_max", "m_tow", "oew",
    "mass_est_mean", "ff_kgs_est_mean", "ff_kgs_est_std",
    "mass_est_std", "tow_est_kg"
]

try_kfold = True

# -------------------------------------------------
# Load data
# -------------------------------------------------
df_features_alt = pd.read_parquet('data/fuel_train_with_alt_10parts.parquet')
df_features_train = pd.read_parquet('data/df_train_best_v0.parquet')

df_features_train = df_features_train[default_cols + ["idx", "ff_kgs", "fuel_kg"]]

# Optional: add trajectory columns
traj_cols = [c for c in df_features_alt.columns if ('vrate' in c)]
feature_cols = default_cols + traj_cols

# Merge trajectory features
df_features_train = df_features_train.merge(
    df_features_alt[["idx"] + traj_cols], on='idx', how='left'
)

# Filter valid training values
df_features_train = df_features_train[
    (df_features_train['ff_kgs'] < 6.5) &
    (df_features_train['ff_kgs'] > 0.05)
]

# Load rank (test) data
df_rank_alt = pd.read_parquet('data/fuel_rank_with_alt_10parts.parquet')
df_features_rank = pd.read_parquet('data/df_rank_best_v0.parquet')
df_features_rank = df_features_rank[default_cols + ["idx", "ff_kgs", "fuel_kg"]]
df_features_rank = df_features_rank.merge(
    df_rank_alt[["idx"] + traj_cols], on='idx', how='left'
)

# -------------------------------------------------
# Settings
# -------------------------------------------------
target_col = "ff_kgs"

base_params = {
    "n_estimators": 7500,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.5,
    "reg_alpha": 0.1,
    "metric": "rmse",
    "random_state": 42
}

categorical_features = ["aircraft_type", "phase"]

# -------------------------------------------------
# Build X / y
# -------------------------------------------------
X = df_features_train[feature_cols].copy()
y = df_features_train[target_col].copy()

cat_feats_actual = [c for c in categorical_features if c in X.columns]
for c in categorical_features:
    X[c] = X[c].astype("category")

# -------------------------------------------------
# Optuna objective (Bayesian tuning)
# -------------------------------------------------
def objective(trial):
    params = base_params.copy()
    params.update({
        "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 2000, 12000, log=True),
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "n_jobs": -1,
    })

    kf = KFold(n_splits=5, shuffle=False)
    oof_preds = np.zeros(len(X))

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = LGBMRegressor(**params, verbose=-1)

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            categorical_feature=cat_feats_actual,
            callbacks=[early_stopping(200)]
        )

        oof_preds[valid_idx] = model.predict(X_valid)

    rmse_oof = mean_squared_error(
        y * X["seg_duration"],
        oof_preds * X["seg_duration"],
        squared=False
    )

    print(f"Trial {trial.number} — RMSE_OOF: {rmse_oof:.5f}")
    return rmse_oof

# -------------------------------------------------
# Run Bayesian tuning
# -------------------------------------------------
study = optuna.create_study(direction="minimize", study_name="lgbm_ff_kgs_rmse_oof")
study.optimize(objective, n_trials=300, show_progress_bar=True)

print("\nBest rmse_oof:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Save all trials
trials_df = study.trials_dataframe()
trials_df.to_csv("optuna_trials_rmse_oof.csv", index=False)

# -------------------------------------------------
# Final params
# -------------------------------------------------
best_params = base_params.copy()
best_params.update(study.best_params)
best_params.update({
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_jobs": -1
})

# -------------------------------------------------
# Train final full model
# -------------------------------------------------
print("\nTraining final FULL models...")

X_test = df_features_rank[feature_cols].copy()
for c in categorical_features:
    X_test[c] = X_test[c].astype("category")

seeds = [46]
preds = []

for seed in seeds:
    final_params = best_params.copy()
    final_params["random_state"] = seed

    model = LGBMRegressor(**final_params)

    model.fit(
        X, y,
        categorical_feature=cat_feats_actual,
        eval_metric="rmse"
    )

    preds.append(model.predict(X_test))

ff_kgs_pred = np.median(np.vstack(preds), axis=0)
df_features_rank["ff_kgs"] = ff_kgs_pred
df_features_rank["fuel_kg"] = ff_kgs_pred * df_features_rank["seg_duration"]

# -------------------------------------------------
# Save local submission
# -------------------------------------------------
submission_df = pd.read_parquet('data/fuel_rank_submission.parquet')
submission_df["fuel_kg"] = df_features_rank["fuel_kg"].values

# Find remote latest version
cmd = ["mc", "ls", "opensky/prc-2025-resourceful-quiver/"]
result = subprocess.run(cmd, capture_output=True, text=True, check=True)
output = result.stdout

versions = re.findall(r"resourceful-quiver_v(\d+)\.parquet", output)
if not versions:
    raise ValueError("No resourceful-quiver_vXXX.parquet files found.")

latest_version = max(map(int, versions))
next_version = latest_version + 1
local_filename = f"data/resourceful-quiver_v{next_version}.parquet"

submission_df.to_parquet(local_filename, index=False)
print(f"\nSaved submission → {local_filename}")
print(submission_df[['idx', 'fuel_kg']].head())

# -------------------------------------------------
# Upload to MinIO
# -------------------------------------------------
cmd_upload = [
    "mc", "cp",
    f"data/resourceful-quiver_v{next_version}.parquet",
    "opensky/prc-2025-resourceful-quiver"
]

subprocess.run(cmd_upload, check=True)
print(f"Uploaded resourceful-quiver_v{next_version}.parquet to opensky/prc-2025-resourceful-quiver/")
