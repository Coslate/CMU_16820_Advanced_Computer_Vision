import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
import scipy.io

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

def calculate_vertical_threshold(bboxes, scale_factor=1.5):
    # Sort bounding boxes by their minr (top row) values
    bboxes_sorted = sorted(bboxes, key=lambda x: x[0])
    
    # Calculate gaps between consecutive bounding boxes
    vertical_gaps = [
        abs(bboxes_sorted[i+1][0] - bboxes_sorted[i][0])  # minr of next box - maxr of current box
        for i in range(len(bboxes_sorted) - 1)]
    
    # Use the median gap as the vertical threshold
    mean_gap = np.mean(vertical_gaps)

    # Compensate for the most of the letter has smaller gaps
    vertical_threshold = scale_factor * mean_gap
    return vertical_threshold

def cluster_bounding_boxes(bboxes, vertical_threshold=10):
    # Sort bounding boxes by their minr (top row) values
    bboxes_sorted = sorted(bboxes, key=lambda x: x[0])
    
    lines = []  # List to hold lists of boxes for each line
    current_line = [bboxes_sorted[0]]  # Start with the first bounding box in the first line

    for i in range(1, len(bboxes_sorted)):
        minr, minc, maxr, maxc = bboxes_sorted[i]
        prev_minr, prev_minc, prev_maxr, prev_maxc = current_line[-1]

        # Check if the current box is within the vertical threshold of the previous box
        if abs(minr - prev_minr) <= vertical_threshold:
            current_line.append(bboxes_sorted[i])
        else:
            # Start a new line
            lines.append(current_line)
            current_line = [bboxes_sorted[i]]

    # Append the last line
    lines.append(current_line)

    return lines    

true_label0 = ['TODOLIST', '1MAKEATODOLIST', '2CHECKOFFTHEFIRST', 'THINGONTODOLIST', '3REALIZEYOUHAVEALREADY', 'COMPLETED2THINGS', '4REWARDYOURSELFWITH', 'ANAP']
true_label1 = ['ABCDEFG', 'HIJKLMN', 'OPQRSTU', 'VWXYZ', '1234567890']
true_label2 = ['HAIKUSAREEASY', 'BUTSOMETIMESTHEYDONTMAKESENSE', 'REFRIGERATOR']
true_label3 = ['DEEPLEARNING', 'DEEPERLEARNING', 'DEEPESTLEARNING']
true_label = [true_label0, true_label1, true_label2, true_label3]
cnt_idx = 0
for img in os.listdir("../images"):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join("../images", img)))
    bboxes, bw = findLetters(im1)

    '''
    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)
    plt.show()
    '''

    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################

    calc_vertical_thresh = calculate_vertical_threshold(bboxes, scale_factor=1.5)
    lines = cluster_bounding_boxes(bboxes, vertical_threshold=calc_vertical_thresh)
    lines_xsorted = [sorted(line, key=lambda x: x[1]) for line in lines]

    '''
    # Verify the line: Draw bounding boxes line by line with different colors
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bw, cmap='gray')  # Assuming 'bw' is your binary image
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

    for i, line in enumerate(lines_xsorted):
        color = colors[i % len(colors)] 
        print(f"i = {i}, len(line) = {len(line)}")
        for minr, minc, maxr, maxc in line:
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')
    plt.show()    
    '''

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    processed_xdata = []

    for i, line in enumerate(lines_xsorted):
        xdata = []
        for j, box in enumerate(line):
            minr, minc, maxr, maxc = box
            cropped_image = bw[minr:maxr, minc:maxc]
            height, width = cropped_image.shape[:2]
            square_size = int(max(height, width)*1.5)

            # Calculate the pad size, pad from central to surrounding boundaries
            top_pad = (square_size - height)//2
            bottom_pad = square_size - height - top_pad
            right_pad = (square_size - width)//2
            left_pad = square_size - width - right_pad

            # Padding
            padded_image = np.pad(cropped_image, pad_width=((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=(1, 1))

            # Resize
            resized_image = skimage.transform.resize(padded_image, (32, 32))
            footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            resized_image = skimage.morphology.erosion(resized_image, footprint)
            
            # Transposing
            transp_image = np.transpose(resized_image)

            # Flatten & Append
            flatten_image = transp_image.flatten()
            xdata.append(flatten_image)

            '''
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
            # Show the original cropped image
            axs[0].imshow(np.zeros((32, 32)), cmap='gray')
            axs[0].set_title(f"Transposed Image {i}_{j}")
            axs[0].axis('off')
        
            # Show the padded image
            axs[1].imshow(resized_image, cmap='gray')
            axs[1].set_title(f"Resized Image {i}_{j}")
            axs[1].axis('off')
        
            # Show the figure and wait for user input
            plt.show()
            '''

        processed_xdata.append(xdata)


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
    )
    params = pickle.load(open("q3_weights.pickle", "rb"))
    ##########################
    ##### your code here #####
    ##########################
    err = 0
    for i, xdata in enumerate(processed_xdata):
        ans_line = []
        for j, x in enumerate(xdata):
            x_test = np.array(x).reshape(1, -1)
            h1_te = forward(x_test, params, "layer1", sigmoid)
            test_probs_te = forward(h1_te, params, "output", softmax)
            detected_letter_idx_te = np.argmax(test_probs_te, axis=1)[0]
            predict_letter = letters[detected_letter_idx_te]
            ans_line.append(predict_letter)
            true_letter = true_label[cnt_idx][i][j]

            #print(f"true_letter = {true_letter}")
            #print(f"predict_letter = {predict_letter}")
            if true_letter != predict_letter:
                print(f"\033[31m{predict_letter}\033[0m", end='')
                err += 1
            else:
                print(f"\033[32m{predict_letter}\033[0m", end='')

            '''
            if i == 4:
                fig, axs = plt.subplots(1, 1, figsize=(10, 5))
                # Show the original cropped image
                axs.imshow(x_test.reshape(32, 32).T, cmap='gray')
                axs.set_title(f"x_test Image")
                axs.axis('off')
                # Show the figure and wait for user input
                plt.show()
                a = input()
            '''
        print(f"")
    print(f"correction rate = {(1-(err/np.sum([len(item) for item in true_label[cnt_idx]])))*100}%")
    cnt_idx += 1
