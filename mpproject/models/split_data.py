import os

import pandas as pd
from data_utils import train_test_split

from mpproject.src.data.path_constants import PREPARED_DATA_PATH

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PREPARED_DATA_PATH, "prepared_data.csv"))
    df["datetime"] = pd.to_datetime(df["datetime"])
    sizes = (0.65, 0.7, 0.8)
    train_test_split(df, sizes)
