import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
# def blur_uv_loss(rgb, inferred_rgb):
#   uv = rgb2uv(rgb)
#   uv_blur0 = rgb2uv(blur(rgb, 3))
#   uv_blur1 = rgb2uv(blur(rgb, 5))

#   inferred_uv = rgb2uv(inferred_rgb)
#   inferred_uv_blur0 = rgb2uv(blur(inferred_rgb, 3))
#   inferred_uv_blur1 = rgb2uv(blur(inferred_rgb, 5))

#   return  ( dist(inferred_uv, uv) +
#             dist(inferred_uv_blur0 , uv_blur0) +
#             dist(inferred_uv_blur1, uv_blur1) ) / 3

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

def loss(actual_image, output):
  actual_uv = get_UV(actual_image)
  output_uv = get_UV(output)
  diff1 = actual_uv - output_uv

  blurred_actual_uv_3 = get_UV(blur(actual_image, 3));
  blurred_output_uv_3 = get_UV(blur(output, 3));
  diff2 = blurred_actual_uv_3 - blurred_output_uv_3

  blurred_actual_uv_5 = get_UV(blur(actual_image, 5));
  blurred_output_uv_5 = get_UV(blur(output, 5));
  diff3 = blurred_actual_uv_5 - blurred_output_uv_5
  
  return (np.linalg.norm(diff1) + np.linalg.norm(diff2) + np.linalg.norm(diff3))/3


# img = cv2.imread('boat.jpg')
# output = blur(img, 4)

# diff = transform_img - img;

# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(transform_img),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()
# diff = transform_img - img;
# print np.linalg.norm(diff)
# print np.mean(transform_img.astype(int));
# print np.mean(img);

# b = get_UV(100, 2, 3);
# print a;
# print convert_RGB(b[0], b[1], b[2]);
# print loss(img, output)