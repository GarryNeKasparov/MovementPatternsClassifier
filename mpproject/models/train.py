import os
from typing import List

import numpy as np
import pytorch_warmup as warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from data_utils import build_loader
from focal_loss.focal_loss import FocalLoss
from models import GRU
from set_seed import seed_everything
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.utils import class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_outputs_and_targets(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> List[np.ndarray]:
    """
    Возвращает предсказания (логиты) по данным.
    """
    model.eval()
    outputs = np.empty((len(loader) * loader.batch_size, 2))
    targets = np.empty(len(loader) * loader.batch_size)
    start = 0
    end = loader.batch_size
    seed_everything(42)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            outputs[start:end] = logits.cpu().numpy()
            targets[start:end] = y.numpy()
            start += loader.batch_size
            end += loader.batch_size
    return outputs, targets.astype(np.int8)


def eval_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    metrics: list,
    device: str = "cuda:0",
) -> dict:
    """
    Возвращает оценку модели по заданным метрикам.
    """
    seed_everything(42)
    model.eval()
    scores = {}
    val_outputs, val_targets = get_outputs_and_targets(model, val_loader, device)
    val_outputs = F.softmax(torch.Tensor(val_outputs), dim=1).argmax(dim=1).numpy()
    for metric in metrics:
        scores[metric.__name__] = metric(val_outputs, val_targets)
    return scores


def eval_loss(
    model: nn.Module, val_loader: DataLoader, criterion, device: str = "cuda:0"
) -> float:
    """
    Возвращает значение loss фунции на данных.
    """
    losses = []
    seed_everything(42)
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y).item()
            losses.append(loss)
    return np.mean(losses)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    criterion,
    scheduler=None,
    warmup_scheduler=None,
    epochs: int = 10,
    val_loader: DataLoader = None,
    compute_val_loss: bool = True,
    compute_metrics: bool = True,
    metrics: list = None,
    verbose: bool = True,
    checkpoint_path: str = "./",
    checkpoint_step: int = 10,
    device: str = "cuda:0",
    wb: bool = False,
) -> List[np.ndarray]:
    """
    Обучает заданную модель на данных.
    """
    scaler = torch.cuda.amp.GradScaler()
    train_history, val_history = [], []
    if compute_val_loss:
        weights = class_weight.compute_class_weight(
            "balanced",
            classes=np.array([0, 1]),
            y=val_loader.dataset.y.numpy(),
        )
        weights = torch.tensor(weights, dtype=torch.float32)
        val_criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        loop = tqdm(train_loader, desc="Training", position=0, leave=True)
        epoch_history = 0

        if epoch % checkpoint_step == 0:
            torch.save(model.state_dict(), checkpoint_path + str(epoch) + ".pt")

        for x, y in loop:
            model.zero_grad(set_to_none=True)
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs = model(x)
                if type(criterion).__name__ == "FocalLoss":
                    outputs = F.softmax(outputs, dim=1)
                loss = criterion(outputs, y)

            epoch_history += loss.item()

            if verbose:
                loop.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                with warmup_scheduler.dampening():
                    scheduler.step()

        if compute_val_loss:
            val_loss = eval_loss(model, val_loader, val_criterion, device)
            if not wb and verbose:
                tqdm.write(f"Val loss: {val_loss:.4f}")
            val_history.append(val_loss)

        if compute_metrics:
            scores = eval_metrics(model, val_loader, metrics, device)
            if not wb and verbose:
                s = ""
                for metric in metrics:
                    s += f"{metric.__name__}: {scores[metric.__name__]:.4f} "
                tqdm.write(s)
        train_history.append(epoch_history / len(train_loader))
        if wb:
            scores["train_loss"] = epoch_history / len(train_loader)
            scores["val_loss"] = val_loss
            wandb.log(scores)

    return train_history, val_history


if __name__ == "__main__":
    params = {
        "hidden_size": 32,
        "num_layers": 2,
        "learning_rate": 1e-3,
        "dropout": [0.5, 0.5],
        "epochs": 3,
    }
    wandb.init(
        project="mpproject",
        config={
            "optimizer": "AdamW",
            "criterion": "FocalLoss",
            "learning_rate": params["learning_rate"],
            "architecture": f'GRU, hidden={params["hidden_size"]}, \
            layers={params["num_layers"]}, dropout={params["dropout"]}',
            "epochs": params["epochs"],
        },
    )

    train_loader, val_loader = build_loader("train"), build_loader("val")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    weights = class_weight.compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=train_loader.dataset.y.numpy()
    )
    weights = torch.tensor(weights, dtype=torch.float32)

    def f1(pred, y):
        return f1_score(y, pred, zero_division=0.0)

    f1.__name__ = "f1_score"

    def matthews(pred, y):
        return matthews_corrcoef(y, pred)

    matthews.__name__ = "matthews_corr"

    model = GRU(
        input_size=train_loader.dataset.X.shape[-1],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropouts=params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
    num_steps = len(train_loader) * 80
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    criterion = FocalLoss(gamma=2, weights=weights.to(device), reduction="mean").to(
        device
    )

    train(
        model,
        train_loader,
        optimizer,
        criterion,
        scheduler=lr_scheduler,
        warmup_scheduler=warmup_scheduler,
        epochs=params["epochs"],
        val_loader=val_loader,
        compute_val_loss=True,
        compute_metrics=True,
        metrics=(accuracy_score, f1, matthews),
        device=device,
        verbose=True,
        checkpoint_path="./checkpoint/",
        checkpoint_step=1000,
        wb=False,
    )
    wandb.finish()
    torch.save(
        model.state_dict(),
        os.path.join("mpproject/models/files/weights/", "GRU_trained.pt"),
    )
