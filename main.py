from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

data_dir = Path('cell-coverage-data')
image_dir = data_dir / 'image'
mask_dir = data_dir / 'mask'
files = sorted(image_dir.glob("*_channel3.tif"))

results = []

def coverage_percent(mask):
    return np.mean(mask) * 100

def predict_mask(image):
    threshold = threshold_otsu(image)
    return  image > threshold


results = []

for path in files:
    image = np.asarray(Image.open(path))
    reference = np.asarray(Image.open(mask_dir / path.name)) > 0

    prediction = predict_mask(image)

    predicted = coverage_percent(prediction)
    actual = coverage_percent(reference)
    error = abs(predicted - actual)

    results.append((path.name, predicted, actual, error))

    print(f"{path.name}: Error = {error:.2f} pp")

mean_error = np.mean([result[3] for result in results])

print(f"\nImages evaluated: {len(results)}")
print(f"Mean absolute error: {mean_error:.2f} percentage points")

worst = max(results, key=lambda result: result[3])

filename = worst[0]

image = np.asarray(Image.open(image_dir / filename))
reference = np.asarray(Image.open(mask_dir / filename)) > 0
prediction = predict_mask(image)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image")

axes[1].imshow(prediction, cmap="gray")
axes[1].set_title("Predicted Mask")

axes[2].imshow(reference, cmap="gray")
axes[2].set_title("Reference Mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("worst_prediction.png", dpi=200)
plt.show()

print(f"Worst image: {filename}")
print(f"Predicted coverage: {worst[1]:.2f}%")
print(f"Reference coverage: {worst[2]:.2f}%")
print(f"Absolute error: {worst[3]:.2f} pp")