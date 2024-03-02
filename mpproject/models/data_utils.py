import multiprocessing as mp
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
from constants import (
    BATCH_SIZE,
    BLOCK_SIZE,
    INPUT_SIZE,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from tqdm.contrib.concurrent import process_map


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
        train_test_split,
        [(df[df.object_id == k], k, 15, sizes) for k in df["object_id"].unique()],
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
    return X_train, y_train, X_calib, y_calib, X_val, y_val, X_test, y_test


class Data(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(
            y, dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_loaders(df, sizes) -> List[DataLoader]:
    """
    Возвращает dataloaders для каждой из частей выборки.
    df : pd.DataFrame - исходные данные.
    sizes : List[float] - размер train, calib, val (test получается автоматически).
    """
    (
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = train_test_split(df, sizes)

    train_data = Data(X_train, y_train)
    calib_data = Data(X_calib, y_calib)
    val_data = Data(X_val, y_val)
    test_data = Data(X_test, y_test)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    calib_loader = DataLoader(
        calib_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, calib_loader, val_loader, test_loader
