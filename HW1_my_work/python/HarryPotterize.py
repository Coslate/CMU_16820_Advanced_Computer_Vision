import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH   import computeH_ransac
from numpy.typing import NDArray
from typing import Tuple, Any
from displayMatch import displayMatched
from planarH import compositeH

# Import necessary functions

# Q2.2.4

def warpImage(opts):
    # Read Input Image
    image_cv_cover      = cv2.imread('../data/cv_cover.jpg')
    image_cv_desk       = cv2.imread('../data/cv_desk.png')
    image_hp_cover      = cv2.imread('../data/hp_cover.jpg')

    # Compute Homography
    matches, locs1, locs2 = matchPics(image_cv_desk, image_cv_cover, opts)
    x1 = locs1[matches[:, 0], :]
    x2 = locs2[matches[:, 1], :]
    x1_correct_pt = np.flip(x1, axis=1)
    x2_correct_pt = np.flip(x2, axis=1)
    H2to1_ransac, inliers = computeH_ransac(x1_correct_pt, x2_correct_pt, opts, fit_inlier_last=True, fit_inlier_last_num=4)

    # Resize the image_hp_cover to the size of image_cv_cover
    resize_image_hp_cover = cv2.resize(image_hp_cover, (image_cv_cover.shape[1], image_cv_cover.shape[0]))

    # Warping hp_cover.jph to cv_desk.png
    warped_hp_image = cv2.warpPerspective(resize_image_hp_cover, H2to1_ransac, (image_cv_desk.shape[1], image_cv_desk.shape[0]))

    # Composite the warped hp_cover.jpg with cv_desk.png
    composite_img = compositeH(H2to1_ransac, resize_image_hp_cover, image_cv_desk)

    cv2.imshow('composite_img', composite_img)
    cv2.waitKey(0)

if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


