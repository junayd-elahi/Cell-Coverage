from pathlib import Path

import numpy as np
from PIL import Image

data_dir = Path('cell-coverage-data')

filename = "AS_09125_050116000001_A24f00d0_slice1_channel3.tif"

image = np.asarray(Image.open(data_dir / "image" /filename))
reference = np.asarray(Image.open(data_dir / "mask" /filename))

def coverage_percent(mask):
    return np.mean(mask) * 100

test_mask = np.array([
    [True, False],
    [False, False]
])

print("Test coverage:", coverage_percent(test_mask))
print("Reference coverage:", coverage_percent(reference))


# print("Image shape: ", image.shape)
# print("Intensity range: ", image.min(), image.max())
# print("Reference shape: ", reference.shape)