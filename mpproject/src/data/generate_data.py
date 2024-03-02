import os
import random
import sys
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pandas as pd
from path_constants import INITIAL_DATA_PATH


def generate_random_dataset(num_rows):
    object_ids = []
    xs = []
    ys = []
    times = []
    velocities = []
    targets = []

    for _ in range(num_rows):
        object_id = random.randint(1, 20)
        object_ids.append(object_id)

        start_date = datetime.now() - timedelta(weeks=random.randint(180, 185))
        random_time = start_date + timedelta(
            minutes=random.randint(0, 10), hours=random.randint(0, 5)
        )
        times.append(random_time.isoformat(sep=" ", timespec="seconds"))

        velocity = round(random.uniform(0, 40), 2)
        velocities.append(velocity)

        x = random.uniform(40 + (object_id * 27 % 20), 43 + (object_id * 27 % 20))
        y = random.uniform(120 + (object_id * 27 % 20), 123 + (object_id * 27 % 20))
        xs.append(x)
        ys.append(y)

        target = np.random.choice(
            [1, 0], p=[0.3 * (x + y) / 100, 1 - 0.3 * (x + y) / 100]
        )
        targets.append(target)

    df = pd.DataFrame(
        {
            "object_id": object_ids,
            "datetime": times,
            "x": xs,
            "y": ys,
            "velocity": velocities,
            "target": targets,
        }
    )

    return df


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Provide number of samples in first argument"
    size = int(sys.argv[1])
    df = generate_random_dataset(size)
    df = df.sort_values(by=["object_id", "datetime"]).reset_index(drop=True)
    idx = np.random.randint(0, size, min(size // 10, 2000))
    df.loc[idx, "velocity"] = np.nan
    idx = np.random.randint(0, size, min(size // 4, 436))
    df.loc[idx, "x"] += 30
    df.loc[idx, "y"] += 20
    df.to_csv(os.path.join(INITIAL_DATA_PATH, "data.csv"), index=False)
