import argparse
import logging
import os
from io import BytesIO
from typing import Dict
from zipfile import BadZipFile, ZipFile

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="raw/dasmalwerk")
    args = parser.parse_args()
    return args


def get_hrefs() -> Dict[str, str]:
    html = requests.get("https://das-malwerk.herokuapp.com").text
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")[1:]
    malware2href = {}
    for row in rows:
        a_tags = row.find_all("a")
        file_hash = a_tags[1].text
        href = a_tags[0]["href"]
        malware2href[file_hash] = href
    return malware2href


def download(malware2href: Dict[str, str], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for file_hash, href in malware2href.items():
        source = requests.get(href, allow_redirects=True)
        try:
            with ZipFile(BytesIO(source.content)) as f:
                f.extractall(path=save_dir, pwd=b"infected")
        except BadZipFile:
            logger.debug(f"Skipping {file_hash}")
            continue


def main(args: argparse.Namespace) -> None:
    malware2href = get_hrefs()
    download(malware2href, args.save_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
