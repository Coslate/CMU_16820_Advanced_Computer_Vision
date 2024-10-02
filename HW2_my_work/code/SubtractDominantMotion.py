import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
import cv2
from scipy.ndimage import median_filter

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance, use_inverse):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    ################### TODO Implement Substract Dominent Motion ###################
    # Calculate the transformation matrix M
    if use_inverse:
        print(f"> Use InverseCompositionAffine()...")
        M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    else:
        print(f"> Use LucasKanadeAffine()...")
        M = LucasKanadeAffine(image1, image2, threshold, num_iters)

    # Warping It
    height, width = image1.shape
    warped_image1 = cv2.warpAffine(image1, M, (width, height), flags=cv2.INTER_LINEAR)

    # Substracting
    diff = cv2.absdiff(image2, warped_image1)

    # Thresholding
    ret, mask = cv2.threshold(diff, tolerance, 255, cv2.THRESH_BINARY)

    # Filter the threshold image to remove noise
    #filtered_image = median_filter(mask, size=3)

    # Morphological operations
    dil1_structure = np.ones((5, 5))
    dil2_structure = np.ones((5, 5))
    ero_structure = np.ones((9, 9))
    mask_dilated = binary_dilation(mask, structure=dil1_structure).astype(np.uint8)    
    mask_eroded = binary_erosion(mask_dilated, structure=ero_structure).astype(np.uint8)
    mask_dilated = binary_dilation(mask_eroded, structure=dil2_structure).astype(np.uint8)    
    mask_dilated = binary_dilation(mask_dilated, structure=dil2_structure).astype(np.uint8)    

    '''
    cv2.imshow("diff", diff*255)
    cv2.imshow("mask", mask)
    cv2.imshow("filtered_image", filtered_image)
    cv2.imshow("mask_eroded", mask_eroded*255)
    cv2.imshow("mask_dilated", mask_dilated*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

#    return mask_dilated.astype(bool)
    return mask_dilated.astype(bool)
