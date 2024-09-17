import numpy as np
import cv2
import skimage.color
from scipy.ndimage import rotate
from opts import get_opts
import matplotlib.pyplot as plt
from helper import briefMatch
from helper import computeBrief
from helper import computeBriefRotInv
from helper import corner_detection
from helper import corner_detectionRotInv
from scipy.signal import convolve2d

# Q2.1.7

def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    image      = cv2.imread('../data/cv_cover.jpg')
    histo      = [0]*36

    for i in range(36):

        # TODO: Rotate Image
        image_rotated = rotate(image, angle=i*10, reshape=True)

        # TODO: Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPicsRotTest(image, image_rotated, opts)
    
        # TODO: Update histogram
        histo[i] = len(matches)

        print(f"progress: {i/36*100}%")

    # TODO: Display histogram
    plt.hist([i*10 for i in range(36)], bins=36, weights=histo)
    plt.xlabel('Rotation')
    plt.ylabel('Number of Matches')
    plt.title('Histogram of Rotation vs Number of Matches')
    plt.show()

def matchPicsRotTest(I1, I2, opts):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """
        
        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
        

        # Convert Images to GrayScale
        # Convert BGR order (from cv2.imread()) to RGB order
        image1_rgb = I1[:, :, [2, 1, 0]]
        image2_rgb = I2[:, :, [2, 1, 0]]
        gray_I1 = skimage.color.rgb2gray(image1_rgb)
        gray_I2 = skimage.color.rgb2gray(image2_rgb)

        # Detect Features in Both Images
        #locs1 = corner_detectionRotInv(gray_I1, sigma);
        #locs2 = corner_detectionRotInv(gray_I2, sigma);
        locs1 = corner_detection(gray_I1, sigma);
        locs2 = corner_detection(gray_I2, sigma);

        # Calculate the orientation of each of the feature points
        orientation_fp_locs1 = calculatePrimalOrientation(locs1, gray_I1)
        orientation_fp_locs2 = calculatePrimalOrientation(locs2, gray_I2)
        
        # Obtain descriptors for the computed feature locations
        desc1, locs1 = computeBriefRotInv(gray_I1, locs1, orientation_fp_locs1)
        desc2, locs2 = computeBriefRotInv(gray_I2, locs2, orientation_fp_locs2)
        #desc1, locs1 = computeBrief(gray_I1, locs1)
        #desc2, locs2 = computeBrief(gray_I2, locs2)

        # Match features using the descriptors
        matches = briefMatch(desc1, desc2, ratio)

        return matches, locs1, locs2

def calculatePrimalOrientation(locs:[], gray_img) -> []:
    orientation_fp = np.zeros(locs.shape[0], dtype=np.float32)

    # Compute Gradient
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0],
                        [1, 2, 1]])
    Gx_img = convolve2d(gray_img, sobel_x, mode='same')
    Gy_img = convolve2d(gray_img, sobel_x, mode='same')

    for i, (x, y) in enumerate(locs):
        # Extract patch around feature point
        Gx_img_size = Gx_img.shape
        Gy_img_size = Gx_img.shape
        patch_size = (9, 9)
        half_w = patch_size[1]//2
        half_h = patch_size[0]//2
        patch_Gx = Gx_img[max(0, y-half_h):min(Gy_img_size[0], y+half_h), max(0, x-half_w): min(Gx_img_size[1], x+half_w)]
        patch_Gy = Gx_img[max(0, y-half_h):min(Gy_img_size[0], y+half_h), max(0, x-half_w): min(Gx_img_size[1], x+half_w)]

        # Calculate the orientation over patch_Gx and patch_Gy
        magnitude = np.sqrt(patch_Gx*patch_Gx + patch_Gy*patch_Gy)
        orientation = (np.degrees(np.arctan2(patch_Gy, patch_Gx)) + 360)%360 # make sure [0, 360]

        # Use historgram to find the primal orientation
        bin_angle = [x*10 for x in range(36)]
        hog, bin_angle = np.histogram(orientation, bins=bin_angle, weights=magnitude)
        max_bin_index = np.argmax(hog)
        most_angle = bin_angle[max_bin_index]
        orientation_fp[i] = most_angle


    return orientation_fp

if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)

