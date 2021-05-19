'''
  SCC0251 - Processamento de Imagens
  Title:
    Filtering
  Authors:
    Marcus Vinícius Medeiros Pará - 11031663
    Gabriel Francischini de Souza - 9361052
  2021/1
'''

import numpy as np
import imageio as io
from enum import Enum
from typing import Union

class Option(Enum):
  FILTER_1D = 1
  FILTER_2D = 2
  MEDIAN = 3

def weights_required(method: Option):
  '''
  Verifies if it's necessary to read the weights of the filter
  '''
  return method != Option.MEDIAN


def read_line() -> np.ndarray:
  '''
  Reads line as np.ndarray of float from stdin
  '''
  return np.array([float(x) for x in input().strip().split(' ')], dtype=np.float64)


def read_weights(method: Option, size: int) -> np.ndarray:
  '''
  Reads the weights of the filter from stdin
  '''
  if method == Option.FILTER_1D:
    return read_line()

  elif method == Option.FILTER_2D:
    return np.array([read_line() for i in range(size)], dtype=np.float64)
  

def filter_1d(src: np.ndarray, filter_size: int, filter_weights: np.ndarray) -> np.ndarray:
  '''
  Returns the filtered src using 1-dimentsional filter_weights as filter
  '''
  side_length: int = len(src)
  dst: np.ndarray = np.zeros(src.size, dtype=np.float64)
  src = np.pad(src.flatten(), filter_size//2, mode="wrap")
  n: int = len(src)
  a: int = filter_size// 2

  for i in range(a, n - a):
    sub_src = src[i - a : i + 1 + a]
    dst[i - a] = np.sum(np.multiply(sub_src, filter_weights))

  dst = dst.reshape(side_length, side_length)

  return dst


def filter_2d(src: np.ndarray, filter_size: int, filter_weights: np.ndarray) -> np.ndarray:
  '''
  Returns the filtered src using 2-dimentsional filter_weights as filter
  '''
  dst: np.ndarray = np.zeros(src.shape, dtype=np.float64)
  src = np.pad(src, filter_size//2, mode="symmetric")
  n, m = src.shape
  a:int = filter_size // 2

  for i in range(a, n - a):
    for j in range(a, m - a):
      sub_src = src[i - a : i + 1 + a, j - a : j + 1 + a]
      dst[i - a][j - a] = np.sum(np.multiply(sub_src, filter_weights))

  return dst


def filter_median(src: np.ndarray, filter_size: int) -> np.ndarray:
  '''
  Returns the filtered src using the median filter with filter_size
  '''
  dst: np.ndarray = np.zeros(src.shape, dtype=np.uint8)
  src = np.pad(src, filter_size//2, mode="constant", constant_values=0)
  n,m = src.shape
  a:int = filter_size // 2

  for i in range(a, n - a):
    for j in range(a, m - a):
      sub_src = sorted(src[i - a : i + 1 + a, j - a : j + 1 + a].flatten())
      dst[i - a][j - a] = sub_src[(filter_size*filter_size)//2]
    
  return dst


def filter_image(image: np.ndarray,
                 method: int,
                 filter_size: int,
                 weights: np.ndarray) -> np.ndarray:
  '''
  Returns filtered img using the filtering method and filter weights when necessary
  '''
  if method == Option.FILTER_1D: return filter_1d(image, filter_size, weights)
  if method == Option.FILTER_2D: return filter_2d(image, filter_size, weights)
  if method == Option.MEDIAN: return filter_median(image, filter_size)


def normalize_image(image: np.ndarray):
  '''
  Returns min-max normalization of image
  '''
  scale: np.float64 = 255.0 / (image.max() - image.min())
  return ((image - image.min())*scale).astype(np.uint8)


def RMSE(src: np.ndarray, dst: np.ndarray) -> np.float64:
  '''
  Returns RMSE between src and dst
  '''
  src = src.astype(np.int64)
  dst = dst.astype(np.int64)
  N,M = src.shape

  return np.sqrt(np.sum((src - dst)**2)/N/M)


def __main__():
  filename: str = input().strip()
  method: Option = Option(int(input()))
  filter_size = int(input())
  weights: Union[np.ndarray, None] = None
  if (weights_required(method)):
    weights = read_weights(method, filter_size)
  src_image: np.ndarray = io.imread(filename)
  dst_image: np.ndarray = filter_image(src_image, method, filter_size, weights)
  dst_image = normalize_image(dst_image)
  error: np.float64 = RMSE(src_image, dst_image)

  print(f"{error:.4f}")

  return


if __name__ == '__main__':
  __main__()