import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm.contrib.concurrent import process_map


def haversine(df) -> pd.Series:
    """
    Вычисляет столбец расстояние Хаверсина.

    Аргументы:
    df : pd.DataFrame - датафрейм.
    """
    object_shift = df["object_id"].shift(1)
    x = df["x"]
    y = df["y"]
    x_shift = x.shift(1)
    y_shift = y.shift(1)

    x = np.radians(x)
    y = np.radians(y)
    x_shift = np.radians(x_shift)
    y_shift = np.radians(y_shift)

    delt_x = x_shift - x
    delt_y = y_shift - y
    delt_x.loc[df["object_id"] != object_shift] = np.nan
    delt_y.loc[df["object_id"] != object_shift] = np.nan

    a = np.square(np.sin(delt_x / 2)) + np.cos(x) * np.cos(x_shift) * np.square(
        np.sin(delt_y / 2)
    )
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c  # радиус Земли
    return km


def compute_velocity(
    df,
    column_name,
    insert_dist=False,
    insert_dtime=False,
) -> pd.DataFrame:
    """
    Вычисляет среднюю скорость за время движения.

    Аргументы:
    df : pd.DataFrame - датафрейм.
    column_name : str - имя столбца посчитанной скорости.
    insert_dtime : bool- вставлять ли столбец расстояний.
    insert_dtime : bool- вставлять ли столбец прошедшего времени.
    """
    delt_time = df["datetime"].diff().apply(lambda x: x.seconds)
    delt_dist = haversine(df)
    df[column_name] = delt_dist / (delt_time / 3600) / 1.852
    if insert_dist:
        df["delt_dist"] = delt_dist
    if insert_dtime:
        df["dtime"] = delt_time
    return df


def build_agg_dict(columns) -> dict:
    """
    Формирует словарь для аггрегирования.

    Аргументы:
    columns : list[str] - массив столбцов данных.
    """
    agg_d = {col: "median" for col in columns if col != "datetime"}
    agg_d["target"] = "max"
    return agg_d


def df_resample_by_object(args: list) -> pd.DataFrame:
    """
    Выполняет перегруппировку датафрейма, относящегося к одному объекту.

    Аргументы:
    args = (k, freq).
    dfd : pd.DataFrame - данные по одному объекту.
    freq : str : шаг группировки (см доступные в pandas).
    """
    dfd, freq, agg_d = args
    dfd = (
        dfd.set_index("datetime")
        .resample(freq)
        .agg(agg_d)
        .dropna(how="all")
        .reset_index()
    )
    return dfd


def df_resample_total(df, freq) -> pd.DataFrame:
    """
    Выполняет перегруппировку датафрейма, относящегося к одному объекту.

    Аргументы:
    df : pd.DataFrame - данные для перегруппировки.
    freq : str - шаг группировки (см. доступные в pandas).
    """
    agg_d = build_agg_dict(df.columns)
    results = process_map(
        df_resample_by_object,
        [(df[df["object_id"] == k], freq, agg_d) for k in df["object_id"].unique()],
        max_workers=mp.cpu_count(),
    )
    df = pd.concat(results)
    df = df.sort_values(by=["object_id", "datetime"]).reset_index(drop=True)
    return df


class DBSCANDetector:
    """
    Детектор аномалий на основе DBSCAN.
    """

    def __init__(self, data: pd.DataFrame, min_samples: int):
        eps = self.compute_eps(data=data, n_neighbors=min_samples)
        self.clf = DBSCAN(eps=eps, metric="haversine", min_samples=min_samples)

    def compute_eps(self, data: pd.DataFrame, n_neighbors: int, plot=False):
        """
        Вычисляет наиболее вероятное значение eps, используя метод
        ближайших соседей.
        """
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="haversine")
        neighbors_fit = neighbors.fit(data)
        distances, _ = neighbors_fit.kneighbors(data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        if plot:
            plt.plot(distances)
        kneedle = KneeLocator(
            range(1, len(distances) + 1),
            distances,
            S=1.0,
            curve="convex",
            direction="increasing",
        )
        return kneedle.knee_y

    def predict(self, X: pd.DataFrame):
        """
        Определяет аномалии в данных.
        """
        self.clf.fit(X)
        return self.clf.labels_ == -1


def df_replace_outliers_by_object(args: list) -> pd.DataFrame:
    """
    Заменяет аномальные значения в данных, относящихся к одному объекту.

    Аргументы:
    args = (k, freq).
    k : int - номер объекта.
    freq : str : шаг оконной функции.
    """
    dfd, w = args
    dfd = dfd.set_index("datetime")
    dfd["x"] = np.deg2rad(dfd["x"])
    dfd["y"] = np.deg2rad(dfd["y"])
    dfd["dx"] = np.abs(dfd["x"] - dfd["x"].rolling(w).median())
    dfd["dy"] = np.abs(dfd["y"] - dfd["y"].rolling(w).median())

    clf = DBSCANDetector(data=dfd[["x", "y"]], min_samples=4)
    out = clf.predict(dfd[["dx", "dy"]])

    dfd.loc[out, ["x", "y"]] = np.nan
    dfd["x"] = np.rad2deg(dfd["x"])
    dfd["y"] = np.rad2deg(dfd["y"])

    dfd["x"] = dfd["x"].interpolate("time", limit_direction="both")
    dfd["y"] = dfd["y"].interpolate("time", limit_direction="both")

    return dfd.drop(["dx", "dy"], axis=1)


def df_replace_outliers_total(df, freq) -> pd.DataFrame:
    """
    Выявляет и исправляет аномальные значения в данных.

    df : pd.DataFrame - данные для выявления аномалий.
    freq : str - шаг оконной функции для рассчета изменений в координатах.
    """
    results = process_map(
        df_replace_outliers_by_object,
        [(df[df["object_id"] == k], freq) for k in df["object_id"].unique()],
        max_workers=mp.cpu_count(),
    )
    df = pd.concat(results)
    df = df.sort_values(by=["object_id", "datetime"]).reset_index()
    return df
