import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

"""
returns the magnitude of the gradient of the image
"""
def comp_grad(im):
    sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)

    grad_magnt = np.sqrt(sobelx**2 + sobely**2)
    return grad_magnt


def harris_corner_detector():
    pass
