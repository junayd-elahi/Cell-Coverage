from pathlib import Path

import numpy as np
from PIL import Image

data_dir = Path('cell-coverage-data')

filename = "AS_09125_050116000001_A24f00d0_slice1_channel1.tif"

image = np.asarray(Image.open(data_dir / "BBBC008_v1_images" /filename))
reference = np.asarray(Image.open(data_dir / "BBBC008_v1_foreground" /filename))

print("Image shape: ", image.shape)
print("Intensity range: ", image.min(), image.max())
print("Reference shape: ", reference.shape)