'''
  SCC0251 - Processamento de Imagens
  Title:
    Geração de Imagens
  Authors:
    Marcus Vinícius Medeiros Pará - 11031663
    Gabriel Francischini de Souza - 9361052

  2021
'''

# imports
import numpy as np
import random
from math import cos, sin, sqrt
from enum import Enum

#BEGIN - enum used for F parameter
class Generators(Enum):
  GENERATOR_1 = 1
  GENERATOR_2 = 2
  GENERATOR_3 = 3
  GENERATOR_4 = 4
  GENERATOR_5 = 5
#END - enum used for F parameter

def random_walk(size: int, seed: int) -> np.ndarray:
  """ 
    Returns a two-dimensional size x size array using the
    random walk algorithm using seed as parameter
  """
  image: np.ndarray = np.zeros((size, size), dtype=np.float32)
  number_of_steps: int = 1 + size**2
  x: int = 0
  y: int = 0
  image[x][y] = 1
  rand: random.Random = random.Random(seed)

  for nth_step in range(number_of_steps):
    dx: int = rand.randint(-1, 1)
    dy: int = rand.randint(-1, 1)
    x,y = (x + dx)%size, (y + dy)%size
    image[x,y] = 1
  
  return image

def generate_image(size: int, F: Generators, **kwargs) -> np.ndarray:
  """
    Returns a two-dimensional size x size array using F parameter.
    kwargs may contain Q (default 0) or S (default 0) values which may be used 
  """
  Q: int = 0
  S: int = 0
  image: np.ndarray
  
  #BEGIN - getting Q and S values, if they exist
  try:
    Q = kwargs["Q"]
  except KeyError:
    pass

  try:
    S = kwargs["S"]
  except KeyError:
    pass
  #END - getting Q and S values, if they exist

  #BEGIN - Selecting function to generate image
  if (F == Generators.GENERATOR_1):
    image = np.array([[(x*y + 2*y)
                       for y in range(size)]
                       for x in range(size)], dtype=np.float32)

  elif (F == Generators.GENERATOR_2):
    image = np.array([[abs(cos(x/Q) + 2*sin(y/Q))
                      for y in range(size)]
                      for x in range(size)], dtype=np.float32)
    
  elif (F == Generators.GENERATOR_3):
    image = np.array([[abs(3*x/Q - (y/Q)**(1/3)) 
                       for y in range(size)] 
                       for x in range(size)], dtype=np.float32)
  
  elif (F == Generators.GENERATOR_4):
    rand = random.Random(S)
    image = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
      for j in range(size):
        image[j][i] = rand.random()

    
  elif (F == Generators.GENERATOR_5):
    image = random_walk(size, S)
  else:
    raise ValueError(f"Invalid Parameters")
  #END - Selecting function to generate image

  return image

def normalize_image(image: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
  """
    Returns the image array normalized between new_min and new_max
  """
  old_max = image.max()
  old_min = image.min()
  scale = (new_max - new_min)/(old_max - old_min)

  image = image - old_min
  image = image * scale
  image = image + new_min

  return image

def sample_image(image: np.ndarray, output_size: int) -> np.ndarray:
  """
    Returns sampled image with output_size
  """
  input_size: int = image.shape[0]
  scale: float = input_size/output_size
  sampled: np.ndarray = np.zeros((output_size, output_size), dtype=np.float32)

  for i in range(0, output_size):
    for j in range(0, output_size):
      sampled[i][j] = image[int(i*scale)][int(j*scale)]

  return sampled

def quantize_image(image: np.ndarray, n_bits: int) -> np.ndarray:
  """
    Returns image as an uint8 array using n_bits most significatives
    for pixel representation
  """
  size: int = image.shape[0]
  image = normalize_image(image, 0, 255)
  image = image.astype(np.uint8)

  #TODO pythonize this
  for i in range(size):
    for j in range(size):
      image[i][j] = (image[i][j] >> (8 - n_bits)) << (8 - n_bits)

  return image

def RSE(image: np.ndarray, reference: np.ndarray) -> float:
  """
    Compares image and reference and returns error between them
  """
  error: float = 0.0
  size: int = len(image)

  for i in range(size):
    for j in range(size):
      error += pow(float(image[i][j]) - float(reference[i][j]), 2)

  return sqrt(error)

def __main__():
  filename = input().rstrip()
  R = np.load(filename)
  C = int(input())
  F = int(input())
  Q = int(input())
  N = int(input())
  B = int(input())
  S = int(input())
  image: np.ndarray = generate_image(C, Generators(F), S=S, Q=Q)
  image = normalize_image(image, 0, 2**16 - 1)
  image = sample_image(image, N)
  image = quantize_image(image, B)
  error = RSE(image, R)
  print("{:.4f}".format(error))


if __name__ == "__main__":
  __main__()