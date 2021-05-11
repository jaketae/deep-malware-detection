import argparse

import torch

import models
from utils import plot_confusion_matrix, train


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MalConvPlus(8, 4096, 128, 32).to(device)
    train(model, train_loader, val_loader, device, "malconv_plus")
    plot_confusion_matrix(model, test_loader, "malconv_plus", device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
