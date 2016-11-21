import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def get_UV(img):
  r = img.take((0,), axis = 2)
  g = img.take((1,), axis = 2)
  b = img.take((2,), axis = 2)

  Y = 0.299 * r + 0.587 * g + 0.114 * b
  U = -0.147 * r - 0.289 * g + 0.436 * b
  V = 0.615 * r - 0.515 * g - 0.100 * b
  return np.concatenate((U, V), 2)

def convert_UV(img):
  r = img.take((0,), axis = 2)
  g = img.take((1,), axis = 2)
  b = img.take((2,), axis = 2)

  Y = 0.299 * r + 0.587 * g + 0.114 * b
  U = -0.147 * r - 0.289 * g + 0.436 * b
  V = 0.615 * r - 0.515 * g - 0.100 * b
  return np.concatenate((Y, U, V), 2)

def convert_RGB(img):
  Y = img.take((0,), axis = 2)
  U = img.take((1,), axis = 2)
  V = img.take((2,), axis = 2)

  r = Y + 1.140*V
  g = Y - 0.395*U - 0.581*V
  b = Y + 2.032*U
  return np.concatenate((r, g, b), 2)

def blur(img, sigma):
  return cv2.GaussianBlur(img, (5, 5), sigma, sigma)
