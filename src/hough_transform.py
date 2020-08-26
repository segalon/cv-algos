# Implementation of Hough Transform for finding lines and circles

import numpy as np
import cv2
import sys


def get_edges(im):
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_grey = cv2.blur(im_grey, (5, 5))
    im_edges = cv2.Canny(im_grey, 100, 200)
    return im_edges


def hough_peaks(H, threshold):
    a, b = np.where(H > np.max(H) - threshold)
    peaks = np.concatenate((a.reshape(-1, 1), b.reshape(-1, 1)), axis=1)
    return peaks


def find_circles(im_path, r_range, threshold=20):
    im = cv2.imread(im_path)
    im_edges = get_edges(im)

    for r in range(r_range[0], r_range[1]):
        H = hough_circles_acc(im_edges, r)
        centers = hough_peaks(H, threshold)
        draw_circles(im, centers, r)

    cv2.imshow("circles", im)
    cv2.waitKey(0)


def find_lines(im_path, threshold=20):
    im = cv2.imread(im_path)
    im_edges = get_edges(im)

    H, thetas, d = hough_lines_acc(im_edges)
    peaks = hough_peaks(H, threshold)
    draw_hough_lines(im, peaks)

    cv2.imshow("lines", im)
    cv2.waitKey(0)


def hough_lines_acc(im):
    im_sz = im.shape[0]
    dmax = np.sqrt(2*im_sz**2).astype(int)

    quant = 180
    white = 255
    thetas = np.linspace(-90, 90, num=quant, dtype=int)

    H = np.zeros((2*dmax, 181))

    e_rows, e_cols = np.where(im == white)

    for y, x in zip(e_rows, e_cols):
        d = x*np.cos(np.deg2rad(thetas)) + y*np.sin(np.deg2rad(thetas))
        # offset by image size for non negative indices
        d += im_sz
        d = d.astype(int)
        # also offset degrees by 90
        H[d, thetas + 90] += 1

    d = np.arange(2*dmax) - im_sz
    return H, thetas, d


def get_line_pts(d, theta, x):
    if theta != 0:
        y = (d - (np.cos(np.deg2rad(theta)) * x)) / (np.sin(np.deg2rad(theta)))
    else:
        return None
    return y


def draw_hough_lines(im, peaks):
    for i in range(0, peaks.shape[0]):
        d_idx, theta_idx = peaks[i]
        d = d_idx - im.shape[0]
        theta = theta_idx - 90

        num_pts = 1000
        x = np.linspace(1, im.shape[1], num_pts)
        y = get_line_pts(d, theta, x)

        # if theta=0
        if y is None:
            # draw a vertical line
            x = np.full(num_pts, d)
            y = np.linspace(0, im.shape[0], num_pts)

        x = x.astype(int)
        y = y.astype(int)

        clr = (0, 255, 0)
        cv2.line(im, (x[1], y[1]), (x[num_pts-1], y[num_pts-1]), clr, thickness=2)
        cv2.imshow("lines", im)


def hough_circles_acc(im, r):
    white = 255
    thetas = np.linspace(0, 360, num=im.shape[1], dtype=int)

    H_sz = list(im.shape)
    H_sz[0] += r
    H_sz[1] += r

    H = np.zeros((H_sz[1], H_sz[0]))

    e_rows, e_cols = np.where(im == white)

    for y, x in zip(e_rows, e_cols):
        a = (x - r * np.cos(np.deg2rad(thetas))).astype(int)
        b = (y + r * np.sin(np.deg2rad(thetas))).astype(int)
        H[a, b] += 1

    return H


def draw_circles(im, peaks, r):
    for i in range(peaks.shape[0]):
        clr = (0, 255, 0)
        a, b = peaks[i]
        cv2.circle(im, (a, b), r, clr, thickness=1)



