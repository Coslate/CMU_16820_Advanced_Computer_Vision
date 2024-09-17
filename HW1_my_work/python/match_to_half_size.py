import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from numpy.typing import NDArray
from typing import Tuple, Any
from displayMatch import displayMatched
from helper import plotMatches

# Import necessary functions
def displayMatched(opts, image1, image2):
    """
    Displays matches between two images

    Input
    -----
    opts: Command line args
    image1, image2: Source images
    """

    matches, locs1, locs2 = matchPics(image1, image2, opts)

    #display matched features
    plotMatches(image1, image2, matches, locs1, locs2)

# Q2.1.7.2

def scaleTest(opts):
    # Read Input Image
    image_cv_cover1      = cv2.imread('../data/cv_cover.jpg')
    image_cv_cover2      = cv2.imread('../data/cv_cover.jpg')

    # Resize to half size
    new_width = int(image_cv_cover1.shape[1] //np.sqrt(2))
    new_height = int(image_cv_cover1.shape[0] //np.sqrt(2))
    resized_image = cv2.resize(image_cv_cover2, (new_width, new_height))

    displayMatched(opts, image_cv_cover1, resized_image)
    #cv2.imshow('composite_img', composite_img)
    #cv2.waitKey(0)

if __name__ == "__main__":

    opts = get_opts()
    scaleTest(opts)