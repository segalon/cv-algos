import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

"""
returns the magnitude of gradient of image, and the derivatives dx, dy
"""
def comp_grads(im):
    dx = np.array([[-1, 0, 1]])
    dy = np.array([-1, 0, 1])
    dy = dy[..., np.newaxis]

    dim_x = convolve2d(im, dx, mode="same")
    dim_y = convolve2d(im, dy, mode="same")

    return dx, dy, np.sqrt(dim_x**2 + dim_y**2)



def harris_corner_detector():
    pass
