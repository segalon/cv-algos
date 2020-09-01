import numpy as np
import cv2
from utils import *
from skimage.feature import peak_local_max
from scipy.signal import convolve2d


def comp_grads(im):
    dx = np.array([[-1, 0, 1]])
    dy = np.array([-1, 0, 1])
    dy = dy[..., np.newaxis]

    dim_x = convolve2d(im, dx, mode="same")
    dim_y = convolve2d(im, dy, mode="same")

    return dim_x, dim_y


def harris_corner_detector(im, alpha=0.04):
    Ix, Iy = comp_grads(im)

    IxIy = np.multiply(Ix, Iy)

    Ix_sq = cv2.GaussianBlur(np.float32(Ix**2), (3, 3), cv2.BORDER_DEFAULT)
    Iy_sq = cv2.GaussianBlur(np.float32(Iy**2), (3, 3), cv2.BORDER_DEFAULT)

    R = np.zeros_like(im)

    # TODO: vectorize if needed
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            det_M = (Ix_sq[y,x] * Iy_sq[y,x]) -2*IxIy[y, x]
            trace_M = Ix_sq[y,x] + Iy_sq[y,x]
            R[y, x] = det_M - alpha * (trace_M**2)

    return peak_local_max(R, min_distance=10)


def draw_corners(im, local_maxes):
    for loc_max in local_maxes:
        y, x = loc_max
        cv2.circle(im, (x, y), 1, (0, 255, 0), thickness=1)

    cv2.imshow("corners", im)
    cv2.waitKey(0)

