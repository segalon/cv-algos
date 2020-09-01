import cv2
from hough_transform import *
import stereo
from panorama import *


def run_disparity_ssd():
    path1 = '../data/pair2-L.png'
    path2 = '../data/pair2-R.png'

    # path1 = '../data/pair0-L.png'
    # path2 = '../data/pair0-R.png'

    l_im = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    r_im = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    d_im = stereo.disparity_ssd(l_im, r_im, W=35)
    d_im = cv2.normalize(d_im, d_im, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imshow("depth", d_im * 4)
    cv2.waitKey(0)


def run_hough():
    im_path0 = '../data/squares.png'
    im_path1 = '../data/coins.png'

    find_circles(im_path1, [20, 25], threshold=10)
    find_lines(im_path1, threshold=80)



path1 = "../data/coins.png"
path2 = "../data/check.bmp"
path3 = "../data/simA.jpg"
im_color = cv2.imread(path2)
im = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)



