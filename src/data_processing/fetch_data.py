"""Download historical match results, goalscorers, and FIFA rankings."""
from pathlib import Path

import requests

from src.utils.config_loader import PROJECT_ROOT, load_config


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def main():
    config = load_config()
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    download(config["data"]["results_url"], raw_dir / "results.csv")
    download(config["data"]["goalscorers_url"], raw_dir / "goalscorers.csv")
    download(config["data"]["fifa_ranking_url"], raw_dir / "fifa_ranking.csv")


if __name__ == "__main__":
    main()
