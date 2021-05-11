import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch import optim
from torch.nn.functional import sigmoid
from tqdm.auto import tqdm


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def set_plt_style():
    plt.rcParams.update(
        {"text.usetex": True, "font.family": "serif", "font.serif": ["cm"],}
    )


def plot_confusion_matrix(model, test_loader, save_title, device, normalize="all"):
    y_true, y_pred = predict(model, test_loader, device)
    conf_mat = confusion_matrix(y_true, y_pred, normalize=normalize)
    axis_labels = ("Benign", "Malware")
    df = pd.DataFrame(conf_mat, index=axis_labels, columns=axis_labels)
    plot = sns.heatmap(df, annot=True, cmap="Blues")
    plot.figure.savefig(os.path.join("imgs", f"{save_title}_conf_mat.png"), dpi=300)
    plt.close(plot.figure)


def plot_roc_curve(models, test_loader, save_title, device):
    fig, ax = plt.subplots()
    ax.grid(linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if isinstance(models, dict):
        for label, model in models.items():
            fpr, tpr, auc_score = _rates_auc(model, test_loader, device)
            ax.plot(fpr, tpr, label=f"{label} ({auc_score:.2f})")
    else:
        fpr, tpr, auc_score = _rates_auc(models, test_loader, device)
        ax.plot(fp, tpr, label=f"{save_title} ({auc_score:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance (0.5)")
    ax.legend(loc="best")
    fig.savefig(os.path.join("imgs", f"{save_title}_roc.png"), dpi=300)
    plt.close(fig)


def _rates_auc(model, test_loader, device):
    y_true, y_pred = predict(model, test_loader, device, apply_sigmoid=True)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


@torch.no_grad()
def predict(model, data_loader, device, apply_sigmoid=False, to_numpy=True):
    model.eval()
    y_true = []
    y_pred = []
    for inputs, labels in tqdm(data_loader, leave=False):
        inputs = inputs.to(device)
        outputs = model(inputs)
        y_true.append(labels)
        y_pred.append(outputs)
    y_true = torch.cat(y_true).to(int)
    if apply_sigmoid:
        y_pred = sigmoid(torch.cat(y_pred))
    else:
        y_pred = (torch.cat(y_pred) > 0).to(int)
    if to_numpy:
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    assert y_true.shape == y_pred.shape
    model.train()
    return y_true, y_pred


def get_accuracy(model, data_loader, device):
    y_true, y_pred = predict(model, data_loader, device, to_numpy=False)
    return 100 * (y_true == y_pred).to(float).mean().item()


def plot_train_history(train_loss_history, val_loss_history, save_title):
    fig, ax = plt.subplots()
    time_ = range(len(train_loss_history))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("BCE Loss")
    ax.grid(linestyle="--")
    ax.plot(time_, train_loss_history, color="blue", label="train loss")
    ax.plot(time_, val_loss_history, color="red", label="val loss")
    ax.legend(loc="best")
    fig.savefig(os.path.join("figures", f"{save_title}_train_history.png"), dpi=300)
    plt.close(fig)


def train(
    model,
    train_loader,
    val_loader,
    device,
    save_title,
    lr=0.001,
    patience=3,
    num_epochs=50,
    verbose=True,
):
    train_loss_history = []
    val_loss_history = []
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    monitor = EarlyStopMonitor(patience)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=patience
    )
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, device, criterion, optimizer)
        train_loss_history.append(train_loss)
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, criterion)
        val_loss_history.append(val_loss)
        if verbose:
            tqdm.write(
                f"Epoch [{epoch}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
        scheduler.step(val_loss)
        if monitor.step(val_loss):
            break
        if len(val_loss_history) == 1 or val_loss < val_loss_history[-2]:
            torch.save(
                model.state_dict(), os.path.join("checkpoints", f"{save_title}.pt"),
            )
    plot_train_history(train_loss_history, val_loss_history, save_title)


def run_epoch(model, data_loader, device, criterion, optimizer=None):
    total_loss = 0
    for inputs, labels in tqdm(data_loader, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


class EarlyStopMonitor:
    def __init__(self, patience, mode="min"):
        assert mode in {"min", "max"}, "`mode` must be one of 'min' or 'max'"
        self.log = []
        self.mode = mode
        self.count = 0
        self.patience = patience

    def step(self, metric):
        if not self.log:
            self.log.append(metric)
            return False
        flag = metric > self.log[-1]
        if flag == (self.mode == "min"):
            self.count += 1
        else:
            self.count = 0
        self.log.append(metric)
        return self.count > self.patience
