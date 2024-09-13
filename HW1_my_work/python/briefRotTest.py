import numpy as np
import cv2
import skimage.color
from scipy.ndimage import rotate
from matchPics import matchPics
from opts import get_opts

#to-delete
import matplotlib.pyplot as plt

#Q2.1.6

def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    image      = cv2.imread('../data/cv_cover.jpg')
    histo      = [0]*36

    for i in range(36):

        # TODO: Rotate Image
        image_rotated = rotate(image, angle=i*10, reshape=True)

        # TODO: Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(image, image_rotated, opts)
    
        # TODO: Update histogram
        histo[i] = len(matches)

    # TODO: Display histogram
    plt.hist([i*10 for i in range(36)], bins=36, weights=histo)
    plt.xlabel('Rotation')
    plt.ylabel('Number of Matches')
    plt.title('Histogram of Rotation vs Number of Matches')
    plt.show()


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
