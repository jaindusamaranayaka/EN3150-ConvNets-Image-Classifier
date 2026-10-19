"""Download the GTSRB training set into data/gtsrb/, the folder every notebook reads.

Run once from the project root:

    python download_data.py

torchvision downloads and extracts the archive (about 180 MB) to data/gtsrb/GTSRB/Training/.
If the images are already there, nothing is downloaded. The script then checks that every
image listed in splits/*.csv exists.
"""

from pathlib import Path

import pandas as pd
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = PROJECT_ROOT / "splits"


def main():

    dataset = datasets.GTSRB(root=DATA_DIR, split="train", download=True)
    print(f"GTSRB ready: {len(dataset):,} images in {DATA_DIR / 'gtsrb' / 'GTSRB' / 'Training'}")

    # The split CSVs list paths relative to data/gtsrb/
    for split in ("train", "val", "test"):
        split_df = pd.read_csv(SPLIT_DIR / f"{split}_split.csv")
        missing = [path for path in split_df.Path if not (DATA_DIR / "gtsrb" / path).exists()]
        print(f"{split:5s} split: {len(split_df):,} images, {len(missing)} missing")

        if missing:
            raise FileNotFoundError(f"{len(missing)} images in {split}_split.csv were not found, e.g. {missing[0]}")


if __name__ == "__main__":
    main()
