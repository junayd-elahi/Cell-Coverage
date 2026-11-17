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
