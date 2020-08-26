import stereo
import cv2
import numpy as np
from hough_transform import *


def run_disparity_ssd():
    path1 = '../images/pair2-L.png'
    path2 = '../images/pair2-R.png'

    # path1 = '../images/pair0-L.png'
    # path2 = '../images/pair0-R.png'

    l_im = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)  # * (1.0 / 255.0)
    r_im = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)  # * (1.0 / 255.0)

    d_im = stereo.disparity_ssd(l_im, r_im, W=35)
    d_im = cv2.normalize(d_im, d_im, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imshow("depth", d_im * 4)
    cv2.waitKey(0)



def run_hough():
    im_path0 = '../images/squares.png'
    im_path1 = '../images/coins.png'

    find_circles(im_path1, [20, 25], threshold=10)
    find_lines(im_path1, threshold=80)


run_disparity_ssd()

