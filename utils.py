import torchvision
import numpy as np
from PIL import Image


def create_meta_from_name(file_name: str):
    feature_names = ["make", "model", "year", "msrp", "front_wheel_size", "hp", 'displacement', 'engine_type',
                     'width', 'height', 'length', 'gas_mileage', 'drivetrain', 'passenger_capacity', 'passenger_doors',
                     'body_style']
    feature_values = file_name.split("_")

    meta = dict(zip(feature_names, feature_values))
    meta["file_name"] = file_name

    return meta


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x.detach().cpu())
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im
