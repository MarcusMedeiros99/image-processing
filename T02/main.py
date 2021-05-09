import numpy as np
import imageio as io
import matplotlib.pyplot as plt
from enum import Enum
from typing import List

class Option(Enum):
  NO_ENHANCEMENT = 1
  INDIVIDUAL_HISTOGRAM_EQUALIZATION = 2
  COLLECTIVE_HISTOGRAM_EQUALIZATION = 3

def calculate_histogram(image: np.ndarray) -> np.ndarray:
  histogram: np.ndarray = np.zeros(256, dtype=np.uint32)

  for line in image:
    for value in line:
      histogram[value] += 1

  return histogram

def normalize_histogram(histogram: np.ndarray, histogram_sum: int) -> np.ndarray:
  normalized_hist: np.ndarray = np.zeros(256, dtype=np.float64)
  histogram = histogram.astype(np.float64)

  for i in range(256): normalized_hist[i] = histogram[i] / histogram_sum

  return normalized_hist

def cumulative_histogram(histogram: np.ndarray) -> np.ndarray:
  cumulative: np.ndarray = np.zeros(len(histogram), dtype=np.float64)
  
  cumulative[0] = histogram[0]
  for i in range(1, 256): cumulative[i] = cumulative[i - 1] + histogram[i]

  return cumulative

def histogram_equalization(image: np.ndarray, cumulative_hist: np.ndarray) -> np.ndarray:
  enhanced_image: np.ndarray = np.zeros(image.shape, dtype=image.dtype)

  for line in image:
    for value in line:
      enhanced_image = np.round(255*cumulative_hist[value])

  return enhanced_image


# TODO
# def collective_histogram_equalization(images: List[np.ndarray], cumulative_hist: np.ndarray) -> List[np.ndarray]:
# def super_resolution(images: List[np.ndarray]) -> np.ndarray:
# def RMSE(image: np.ndarray, reference: np.ndarray) -> float:


def __main__():
  filename = input()

  image = io.imread(filename)
  histogram = calculate_histogram(image)
  histogram = normalize_histogram(histogram, len(image)*len(image[0]))
  cumulative_hist = cumulative_histogram(histogram)
  image = histogram_equalization(image, cumulative_hist)
  # print(cumulative_hist)
  # plt.xlim(0, 255)
  # plt.bar(range(256), cumulative_hist)
  # plt.show()

if __name__ == "__main__":
  __main__()