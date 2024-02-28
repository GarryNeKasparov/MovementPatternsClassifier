import os

import pandas as pd
from setup import (
    INITIAL_DATA_PATH,
    PREPARED_DATA_PATH,
)
from utils import (
    compute_velocity,
    df_replace_outliers_total,
    df_resample_total,
)

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(INITIAL_DATA_PATH, "data.csv"))
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df_resample_total(df, "15min")
    df = df_replace_outliers_total(df, "1d")

    max_v = round(df["velocity"].quantile(0.99))
    df = compute_velocity(df, "velocity_c")

    idx = ((df["velocity"].isna()) | (df["velocity"] > max_v)) & (
        df["velocity_c"] <= max_v
    )
    df.loc[idx, "velocity"] = df.loc[idx, "velocity_c"]
    df["velocity"] = df["velocity"].interpolate()

    df.to_csv(os.path.join(PREPARED_DATA_PATH, "prepared_data.csv"))
