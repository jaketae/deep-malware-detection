import argparse
import os

import models
import torch
from dataset import make_loaders
from utils import set_seed, train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="MalConvPlus")
    parser.add_argument("--embed_dim", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benign_dir", type=str, required=True)
    parser.add_argument("--malware_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="exp1")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="../../assets/checkpoints"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()
    return args


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    model_cls = getattr(models, args.model)
    model = model_cls(
        args.embed_dim, args.max_len, args.out_channels, args.window_size, args.dropout
    ).to(device)
    train_loader, val_loader, _ = make_loaders(
        args.benign_dir,
        args.malware_dir,
        args.batch_size,
        args.val_size,
        args.test_size,
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(model, train_loader, val_loader, device, args.checkpoint_dir, args.tag)


if __name__ == "__main__":
    args = get_args()
    main(args)
