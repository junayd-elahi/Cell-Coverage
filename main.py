from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

data_dir = Path('cell-coverage-data')

filename = "AS_09125_050116000001_A24f00d0_slice1_channel3.tif"

image = np.asarray(Image.open(data_dir / "image" /filename))
reference = np.asarray(Image.open(data_dir / "mask" /filename))

def coverage_percent(mask):
    return np.mean(mask) * 100

def predict_mask(image):
    threshold = threshold_otsu(image)
    return  image > threshold

test_mask = np.array([
    [True, False],
    [False, False]
])

prediction = predict_mask(image)


print("Predicted coverage:", coverage_percent(prediction))

fig, axes = plt.subplots(1,2, figsize=(10,5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original image")

axes[1].imshow(prediction, cmap='gray')
axes[1].set_title("Predicted mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()