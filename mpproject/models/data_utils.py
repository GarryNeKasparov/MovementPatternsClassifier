import multiprocessing as mp
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from tqdm.contrib.concurrent import process_map

from mpproject.models.constants import (
    BATCH_SIZE,
    BLOCK_SIZE,
    INPUT_SIZE,
)
from mpproject.models.models import (
    GRU,
    LSTM,
)


def distance(x: np.ndarray) -> np.ndarray:
    """
    Рассчитывает расстояния от стартовой точки.
    """
    return np.sqrt(np.sum((x[0] - x) ** 2, axis=1))


def get_new_coords(coords: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Формула перехода к новому базису.
    """
    A = np.linalg.inv(basis)
    return np.asarray([A @ v for v in (coords[0:] - coords[0])])


def get_path_pattern(x: np.ndarray) -> np.ndarray:
    """
    Рассчитывает базис и возвращает координаты точек в нем.
    """
    if isinstance(x, list):
        x = np.asarray(x, dtype=np.float32)
    elif not isinstance(x, np.ndarray):
        raise TypeError("x must be a list or a numpy array")
    if len(x) == 0:
        return np.asarray([0, 0])
    # print(x)
    max_d = np.argmax(distance(x))
    v1 = x[max_d] - x[0]
    if np.linalg.norm(v1) == 0:
        v1 = np.asarray([1, 0])
        v2 = np.asarray([0, 1])
    else:
        v1 /= np.linalg.norm(v1)
        v2 = np.asarray([-v1[1], v1[0]])
    basis = np.stack([v1, v2]).T
    pattern = get_new_coords(x, basis)
    return pattern


def to_context(
    X: np.ndarray, y: np.ndarray, block_size, input_size, **kwargs
) -> List[np.ndarray]:
    """
    Рассчитывает паттерн движения. Для этого объединяет наблюдения в группы
    фиксированного размера и с одинаковыми временными метками, если это возможно.

    Далее рассчитывает наибольшее отклонение от начальной точки - это направление
    выбирается в качестве одного вектора в новом базисе. Второй получается автоматически.
    Начальные точки переводятся в новый базис.
    """
    if X.shape[0] <= block_size:
        warnings.warn(
            f"Too few samples. Length of X is {X.shape[0]}, \
            block_size = {block_size}, \
            object_id = {kwargs['object_id']:0.0f}",
            stacklevel=1,
        )
        return np.array([]), np.array([])
    seqs = np.zeros((X.shape[0] - block_size, block_size + 1, input_size))
    targets = np.empty(X.shape[0] - block_size)
    k = 0
    i = 1
    while i < X.shape[0] - block_size:
        end_idx = i + block_size + 1
        if end_idx >= X.shape[0]:
            break
        checker = np.argwhere(
            np.diff(X[i:end_idx, 0])
            > pd.to_timedelta(kwargs["freq"], unit="min")
            + pd.to_timedelta(np.log10(kwargs["freq"]), unit="min")
        )
        if checker.size == 0:
            slc = X[i:end_idx, 1:].astype(np.float32)
            new_coords = get_path_pattern(slc[:, :2])
            seqs[k] = np.hstack([new_coords, slc[:, 2:]])
            targets[k] = y[end_idx - 1]
            k += 1
            i += 1
        else:
            i += checker[-1, 0] + 1
    if k == 0:
        warnings.warn(
            f"Can't construct contexts for given freq. \
            Object_id = {kwargs['object_id']:0.0f}, \
            freq = {kwargs['freq']}.",
            stacklevel=1,
        )
    return seqs[:k], targets[:k]


def train_test_split_by_object(args) -> List[np.ndarray]:
    """
    Разделяет данные на обучающую, валидационную, калибрационную и
    тестовую выборки по одному объекту.
    args : list = (df, object_id, freq, sizes)
    df : pd.DataFrame - данные по одному объекту.
    object_id - номер объекта.
    freq - временной шаг сбора контекста.
    sizes : List[float] - размер train, calib, val (test получается автоматически).
    """
    df, object_id, freq, sizes = args
    X, y = (
        df[["datetime", "x", "y", "velocity"]].values,
        df["target"].values,
    )
    X, y = to_context(X, y, BLOCK_SIZE, INPUT_SIZE, object_id=object_id, freq=freq)
    train, calib, val = (
        int(sizes[0] * len(X)),
        int(sizes[1] * len(X)),
        int(sizes[2] * len(X)),
    )
    X_train, X_calib, X_val, X_test = np.split(X, [train, calib, val])
    y_train, y_calib, y_val, y_test = np.split(y, [train, calib, val])
    return X_train, y_train, X_calib, y_calib, X_val, y_val, X_test, y_test


def train_test_split(df, sizes) -> List[np.ndarray]:
    """
    Разделяет данные на обучающую, валидационную, калибрационную и тестовую выборки.
    df : pd.DataFrame - исходные данные.
    sizes : List[float] - размер train, calib, val (test получается автоматически).
    """
    results = process_map(
        train_test_split_by_object,
        [(df[df.object_id == k], k, 60, sizes) for k in df["object_id"].unique()],
        max_workers=mp.cpu_count(),
    )
    X_train, y_train = np.concatenate(
        [r[0] for r in results if len(r[0].shape) == 3]
    ), np.concatenate([r[1] for r in results])
    X_calib, y_calib = np.concatenate(
        [r[2] for r in results if len(r[2].shape) == 3]
    ), np.concatenate([r[3] for r in results])
    X_val, y_val = np.concatenate(
        [r[4] for r in results if len(r[4].shape) == 3]
    ), np.concatenate([r[5] for r in results])
    X_test, y_test = np.concatenate(
        [r[6] for r in results if len(r[6].shape) == 3]
    ), np.concatenate([r[7] for r in results])
    assert len(X_train) > 0, "train is empty!"
    assert len(X_calib) > 0, "calib is empty!"
    assert len(X_val) > 0, "val is empty!"
    assert len(X_test) > 0, "test is empty!"
    for arr, name in [
        (X_train, "X_train.npy"),
        (X_calib, "X_calib.npy"),
        (X_val, "X_val.npy"),
        (X_test, "X_test.npy"),
        (y_train, "y_train.npy"),
        (y_calib, "y_calib.npy"),
        (y_val, "y_val.npy"),
        (y_test, "y_test.npy"),
    ]:
        np.save(os.path.join("mpproject/models/files/splits", name), arr)


class Data(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(
            y, dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_loader(split) -> List[DataLoader]:
    """
    Возвращает dataloader для указанной части выборки.
    split : str - train/calib/val/test.
    """
    assert split in {
        "train",
        "calib",
        "val",
        "test",
    }, 'Split must be one of "train", "calib", "val", "test"'
    X, y = np.load(
        os.path.join("mpproject/models/files/splits", f"X_{split}.npy")
    ), np.load(os.path.join("mpproject/models/files/splits", f"y_{split}.npy"))
    data = Data(X, y)
    loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def get_model(name) -> torch.nn.Module:
    """
    Возвращает обученную модель.
    name : str - LSTM/GRU.
    """
    assert name in {"LSTM", "GRU"}, 'Unknown models name. Must be one of "LSTM" or "GRU"'
    assert os.path.exists(
        os.path.join("mpproject", "models", "files", "weights", f"{name}_trained.pt")
    ), f"{name} model is not trained yet."
    if name == "GRU":
        model = GRU(3, 32, 2, [0.5, 0.5])
    elif name == "LSTM":
        model = LSTM(3, 32, 2, [0.5, 0.5])
    else:
        raise AssertionError("Unknown model")
    model.load_state_dict(
        torch.load(
            os.path.join("mpproject", "models", "files", "weights", f"{name}_trained.pt")
        )
    )
    return model
