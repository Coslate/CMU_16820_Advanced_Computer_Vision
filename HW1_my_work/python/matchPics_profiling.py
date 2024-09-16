import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper_profiling import computeBrief
from helper import corner_detection
import time

# Q2.1.4

def matchPics(I1, I2, opts, time_crn_dt, time_brief, time_match, i, report_str: []):
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
        gray_I1 = skimage.color.rgb2gray(image1_rgb)
        gray_I2 = skimage.color.rgb2gray(image2_rgb)

        # TODO: Detect Features in Both Images
        start_crn_dt = time.perf_counter()
        locs1 = corner_detection(gray_I1, sigma);
        locs2 = corner_detection(gray_I2, sigma);
        end_crn_dt = time.perf_counter()
        time_crn_dt = end_crn_dt - start_crn_dt
        
        
        # TODO: Obtain descriptors for the computed feature locations
        start_brief = time.perf_counter()
        desc1, locs1 = computeBrief(gray_I1, locs1, i, report_str)
        desc2, locs2 = computeBrief(gray_I2, locs2, i, report_str)
        end_brief = time.perf_counter()
        time_brief = end_brief - start_brief
        

        # TODO: Match features using the descriptors
        start_match = time.perf_counter()
        matches = briefMatch(desc1, desc2, ratio)
        end_match = time.perf_counter()
        time_match = end_match - start_match

        return matches, locs1, locs2, time_crn_dt, time_brief, time_match

