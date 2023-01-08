import argparse
import logging
import os
import pickle

import pefile

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    for file_name in os.listdir(args.input_dir):
        file_path = os.path.join(args.input_dir, file_name)
        try:
            file = pefile.PE(file_path)
            header = list(file.header)
            with open(os.path.join(args.output_dir, f"{file_name}.pickle"), "wb") as f:
                pickle.dump(header, f)
        except pefile.PEFormatError:
            logger.debug(f"Skipping {file_path}")
            continue


if __name__ == "__main__":
    args = get_args()
    main(args)
