# Source: https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper

import os
import json

from datasets import load_dataset, Dataset
from utils import create_meta_from_name

DATA_DIR = "data"
DATA_FOLDER = "thecarconnectionpicturedataset"
TRAIN_RATIO = 0.8


def create_dataset(path: str) -> Dataset:
    """Create a dataset from the data in the given path"""
    # List all files in the directory
    files = os.listdir(path)

    # Create meta data
    meta = [create_meta_from_name(f) for f in files]

    # Save as json lines
    with open(os.path.join(path, "metadata.jsonl"), "w") as outfile:
        for entry in meta:
            json.dump(entry, outfile)
            outfile.write('\n')

    # Load dataset
    data = load_dataset(
        path="imagefolder", name="car_connection_picture", data_dir=os.path.join(DATA_DIR, DATA_FOLDER),
    )

    return data


if __name__ == "__main__":

    car_dataset = create_dataset(os.path.join(DATA_DIR, DATA_FOLDER))

    # Save dataset locally
    car_dataset.save_to_disk(os.path.join(DATA_DIR, "car_connection_picture"))
