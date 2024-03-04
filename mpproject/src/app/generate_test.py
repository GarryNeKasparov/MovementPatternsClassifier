import random
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pandas as pd


def gen_data(num_rows):
    start_date = datetime.now() - timedelta(weeks=random.randint(180, 185))
    start_date = start_date.isoformat(sep=" ", timespec="seconds")
    dates = pd.date_range(start=start_date, periods=num_rows, freq="15T")
    x = np.cos(
        np.linspace(0, np.random.uniform(0, 2 * np.pi, 1)[0], num_rows)
    ) * np.random.randint(28, 30) + np.random.uniform(3, 10, num_rows)
    y = np.sin(
        np.linspace(0, np.random.uniform(0, np.pi / 2, 1)[0], num_rows)
    ) * np.random.randint(38, 40) + np.random.uniform(3, 10, num_rows)
    velocity = np.round(np.random.uniform(0, 40, num_rows), 2) + np.random.uniform(
        0, 10, num_rows
    )
    target = np.random.choice([0, 1], num_rows, p=(0.3, 0.7))
    return pd.DataFrame(
        {
            "object_id": np.random.randint(1, 20, 1)[0],
            "datetime": dates,
            "x": x,
            "y": y,
            "velocity": velocity,
            "target": target,
        }
    )
