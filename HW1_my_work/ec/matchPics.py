import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4

def matchPics(I1, I2, opts, down_sample_factor: int):
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
