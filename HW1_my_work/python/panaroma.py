import numpy as np
import cv2

# Import necessary functions
from opts import get_opts
from typing import List, Any, Tuple
from displayMatch import displayMatched
from planarH import computeH_ransac
from matchPics import matchPics
import os
import time
import shutil
import skimage.color
import numpy.typing as npt



# Q4

#########################
#     Main-Routine      #
#########################
def main():
    # Input arguments/Load input videos
    opts = get_opts()
    out_folder = os.path.dirname(opts.output_file)
    if not os.path.exists(out_folder): 
        os.makedirs(out_folder)
    else: 
        #shutil.rmtree(out_folder)
        #os.makedirs(out_folder)
        pass

    image_left      = cv2.imread(opts.input_left_img_file)
    image_right     = cv2.imread(opts.input_right_img_file)

    # Process each frame
    generatePanorama(image_right, image_left, opts)

#########################
#     Sub-Routine       #
#########################
def generatePanorama(img1: npt.NDArray[Tuple[Any, Any]], img2: npt.NDArray[Tuple[Any, Any]], opts) -> None:
    down_sample_factor = 1

    # Start of the Computation
    # Compute Features Matching
    matches, locs1, locs2 = matchPicsPanaroma(img1, img2, opts, down_sample_factor)
    #matches, locs1, locs2 = matchPics(img1, img2, opts)
    x1 = locs1[matches[:, 0], :]
    x2 = locs2[matches[:, 1], :]
    x1_correct_pt = np.flip(x1, axis=1)
    x2_correct_pt = np.flip(x2, axis=1)

    # Compute Homography
    H2to1_ransac, inliers = computeH_ransac(x1_correct_pt, x2_correct_pt, opts, True, 10)

    # Warp & Composite img2 to frames_book
    composite_img = compositeHPanorama(H2to1_ransac, img2, img1)

    #cv2.imshow("Panorama Result", composite_img)
    #cv2.waitKey(0)
    cv2.imwrite(f"{opts.output_file}", composite_img)

def compositeHPanorama(H2to1, template, img):
    composite_width  = template.shape[1] + img.shape[1]
    composite_height = img.shape[0]
    composite_img    = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)

    # TODO: Create mask of same size as template
    mask = np.full((template.shape[0], template.shape[1], 1), 255, dtype=np.uint8)

    # TODO: Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H2to1, (composite_width, composite_height))

    # TODO: Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H2to1, (composite_width, composite_height))

    # TODO: Use mask to combine the warped template and the image
    '''
    inverse_mask    = cv2.bitwise_not(warped_mask)
    masked_template = cv2.bitwise_and(warped_template, warped_template, mask=warped_mask)
    masked_img      = cv2.bitwise_and(img, img, mask=inverse_mask)
    composite_img   = cv2.add(masked_template, masked_img)
    '''
    last_column = -1
    for i in range(composite_img.shape[0]):
        for j in range(composite_img.shape[1]):
            if warped_mask[i, j] == 255:
                composite_img[i, j] = warped_template[i, j]
                last_column = max(j, last_column)
            else:
                if i < img.shape[0] and j < img.shape[1]:
                    composite_img[i, j] = img[i, j]
                    last_column = max(j, last_column)

    # Crop the image to remove blank areas
    composite_img = composite_img[:, :last_column+1]

    return composite_img

def matchPicsPanaroma(I1, I2, opts, down_sample_factor: int):
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


        # TODO: Convert Images to GrayScale
        # Convert BGR order (from cv2.imread()) to RGB order
        image1_rgb = I1[:, :, [2, 1, 0]]
        image2_rgb = I2[:, :, [2, 1, 0]]
        gray_I1 = (skimage.color.rgb2gray(image1_rgb) * 255).astype('uint8')
        gray_I2 = (skimage.color.rgb2gray(image2_rgb) * 255).astype('uint8')

        # Scale down the image
        gray_I1 = cv2.resize(gray_I1, (gray_I1.shape[1] // down_sample_factor, gray_I1.shape[0] // down_sample_factor))
        gray_I2 = cv2.resize(gray_I2, (gray_I2.shape[1] // down_sample_factor, gray_I2.shape[0] // down_sample_factor))


        #------------------------------Corner Detection----------------------------------#

        # TODO: Detect Features in Both Images
        #locs1 = corner_detection(gray_I1, sigma);
        #locs2 = corner_detection(gray_I2, sigma);

        orb = cv2.ORB_create(nfeatures=20000)
        orb_locs1, orb_desc1 = orb.detectAndCompute(gray_I1, None);
        orb_locs2, orb_desc2 = orb.detectAndCompute(gray_I2, None);

        orb_locs1_coord = np.array([(int(x.pt[1]), int(x.pt[0])) for x in orb_locs1])
        orb_locs2_coord = np.array([(int(x.pt[1]), int(x.pt[0])) for x in orb_locs2])

        #print(f"locs1 = {locs1}")
        #print(f"locs2 = {locs2}")
        #print(f"locs1.shape = {locs1.shape}")
        #print(f"locs2.shape = {locs2.shape}")

        #print(f"orb_locs1_coord = {orb_locs1_coord}")
        #print(f"orb_locs2_coord = {orb_locs2_coord}")
        #print(f"orb_locs1_coord.shape = {orb_locs1_coord.shape}")
        #print(f"orb_locs2_coord.shape = {orb_locs2_coord.shape}")
        
        #------------------------------Descriptor----------------------------------#
        # TODO: Obtain descriptors for the computed feature locations
        #desc1, locs1 = computeBrief(gray_I1, locs1)
        #desc2, locs2 = computeBrief(gray_I2, locs2)
        

        #------------------------------Matching----------------------------------#
        # TODO: Match features using the descriptors
        #matches = briefMatch(desc1, desc2, ratio)
        # FLANN parameters for ORB
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH (for ORB)
                        table_number=12,  # 12
                        key_size=20,     # 20
                        multi_probe_level=2)  # 2
        search_params = dict(checks=10000)  # Specify the number of times the trees in the index should be recursively traversed.

        # Create the FLANN matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Perform matching
        matches = flann.knnMatch(orb_desc1, orb_desc2, k=2)

        # Apply ratio test to keep the best matches
        good_matches = []
        for match in matches:
            if len(match) == 2: #avoid the crashed case that less than 2 matches were found.
                if match[0].distance < 0.75 * match[1].distance:
                     good_matches.append(match[0])
        matches_orb = good_matches
        
        # Create a BFMatcher object with default parameters (L2 norm for ORB)
        #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        #matches_orb = bf.match(orb_desc1, orb_desc2)        
        matches_orb_idx1 = np.array([match.queryIdx for match in matches_orb])
        matches_orb_idx2 = np.array([match.trainIdx for match in matches_orb])
        matches_orb_idx = np.stack((matches_orb_idx1, matches_orb_idx2), axis=1)

        #print(f"matches.shape = {matches.shape}")
        #print(f"matches_orb_idx.shape = {matches_orb_idx.shape}")
        #print(f"matches = {matches}")
        #rint(f"matches_orb_idx = {matches_orb_idx}")

        matches = matches_orb_idx
        locs1 = orb_locs1_coord
        locs2 = orb_locs2_coord

        return matches, locs1, locs2

#---------------Execution---------------#
if __name__ == '__main__':
    main()