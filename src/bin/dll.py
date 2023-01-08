import argparse
import logging
import os
import random
from io import BytesIO
from zipfile import BadZipFile, ZipFile

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download .dll files")
    parser.add_argument(
        "--num_files", type=int, default=100, help="number of files to download"
    )
    parser.add_argument(
        "--save_dir", type=str, default="raw/dll", help="directory to save files"
    )
    args = parser.parse_args()
    return args


def get_href(index: int) -> str:
    link = f"https://wikidll.com/download/{index}"
    html = requests.get(link, allow_redirects=True).text
    soup = BeautifulSoup(html, "html.parser")
    href = soup.find("a", {"class": "download__link"})["href"]
    return href


def download(index: int, save_dir: str) -> None:
    href = get_href(index)
    source = requests.get(href, allow_redirects=True)
    try:
        with ZipFile(BytesIO(source.content)) as f:
            f.extractall(path=save_dir)
    except BadZipFile:
        logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    indices = random.sample(range(1, 27786), args.num_files)
    for index in indices:
        download(index, args.save_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
