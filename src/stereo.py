import numpy as np
import cv2


def disparity_ssd(l_im, r_im, W, pad=False):
    offset = int(W / 2)
    d_im = np.zeros_like(l_im)
    l_im = np.pad(l_im, (offset, offset))
    r_im = np.pad(r_im, (offset, offset))

    i = 0
    for y in range(offset, l_im.shape[0] - offset):
        for xl in range(offset, l_im.shape[1] - offset):
            template = l_im[y - offset: y + offset, xl - offset: xl + offset]

            print_every = 3000
            if i % print_every == 0:
                print(i*100 / (l_im.shape[0] * l_im.shape[1]))
            i += 1

            r_im_strip = r_im[y - offset: y + offset + 1, :]
            min_ssd = cv2.matchTemplate(r_im_strip, template, method=cv2.TM_SQDIFF_NORMED)
            _, _, min_idx, _ = cv2.minMaxLoc(min_ssd)
            d_im[y-offset, xl-offset] = np.abs(xl - min_idx[0])
    return d_im

