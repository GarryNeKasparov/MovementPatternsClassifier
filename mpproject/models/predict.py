import numpy as np
import torch
import torch.nn.functional as F

from mpproject.models.data_utils import get_path_pattern


def get_predictions(model, df, threshold=0.5) -> np.ndarray:
    """
    Для каждой точки из входных данных создает контекст не больше, чем
    из BLOCK_SIZE точек. Используется для предсказания.
    df : pd.DataFrame - входные данные (чистые).
    threshold : float - порог определения положительного класса.
    """
    preds = []
    for i in range(1, df.shape[0] + 1):
        slc = df.iloc[:i]
        pattern = get_path_pattern(slc[["x", "y"]].values)
        feature = torch.Tensor(
            np.hstack((pattern, slc[:i]["velocity"].values.reshape(-1, 1))).reshape(
                1, -1, 3
            )
        )
        pred = model(feature)
        pred = F.softmax(pred, dim=1)[0][1] > threshold
        preds.append(int(pred))
    return preds
