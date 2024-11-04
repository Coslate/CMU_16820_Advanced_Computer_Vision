import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    # To float
    image_float = skimage.img_as_float(image)

    # Blur/Denoise
    image_gau = skimage.filters.gaussian(image_float, sigma=2.3, channel_axis=2)

    # Grayscale
    image_gray = skimage.color.rgb2gray(image_gau)

    # Threshold
    thresh_val = skimage.filters.threshold_otsu(image_gray)
    binary_image = image_gray < thresh_val # Number is white, background is black, for skimage.segmentation.clear_border()

    # Morphological Closing: Dilation + Erosion
    bw_cl = skimage.morphology.closing(binary_image, skimage.morphology.octagon(2, 4))

    # Clearing: As in tutorial, clear white component toudhing border
    bw = skimage.segmentation.clear_border(bw_cl)
    bw = skimage.img_as_float(bw)

    # Invert to have number in black and background in white as required
    bw = 1.0 - bw

    # Label image regions
    labels = skimage.measure.label(bw, connectivity=2, background=1)

    for region in skimage.measure.regionprops(labels):
        # take regions with large enough areas
        if region.area >= 600:
            # draw rectangle around segmented coins
            bboxes.append(region.bbox)

    '''
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.imshow(bw, cmap='gray')
    axs.set_title(f"binary_image Image")
    axs.axis('off')
    plt.show()
    a = input()
    '''
    return bboxes, bw
