import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import cv2

# write your script here, we recommend the above libraries for making your animation
'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''

#########################
#     Main-Routine      #
#########################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1e-2,
        help='dp threshold of Lucas-Kanade for terminating optimization',
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.2,
        help='binary threshold of intensity difference when computing the mask',
    )
    parser.add_argument(
        '--seq_file',
        default='../data/antseq.npy',
    )

    parser.add_argument(
        '--output_folder',
        type=str,
        default='.',
        help='output folder for antseq_frame*.png',
    )

    parser.add_argument(
        '--use_inverse',
        type=int,
        default=0,
        help='whether to use inverse_compositional Lucas Kanade algorithm. Set 0 to use original forwarding Lucas Kanade algorithm.',
    )

    args = parser.parse_args()
    num_iters = args.num_iters
    threshold = args.threshold
    tolerance = args.tolerance
    seq_file = args.seq_file
    output_folder = args.output_folder
    use_inverse = args.use_inverse
    seq = np.load(seq_file)

    for i in [30, 60, 90, 120]:
        if i == (int(seq.shape[2])-1):
            break

        # Motion Detection
        It  = seq[:, :, i]
        It1 = seq[:, :, i+1]
        mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance, use_inverse)

        # Visualize the results
        binary_mask = (mask.copy().astype(np.uint8))*255
        blue_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
        blue_mask[:, :, 2] = binary_mask  # Set the blue channel
        overlay_image = addWeighted(It1, blue_mask, alpha=1, beta=1, mask=mask)

        '''
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.title(f"Frame i: {i}")
        plt.imshow(It1, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title(f"Motion Mask at Frame {i}")
        plt.imshow(blue_mask)  # Superimpose motion mask

        plt.subplot(1, 3, 3)
        plt.title(f"Overlaid Image at Frame {i}")
        plt.imshow(overlay_image)  # Superimpose motion mask
        plt.show()
        '''

        fig, ax = plt.subplots()
        ax.axis('off')
        plt.imsave(f'{output_folder}/antseq_frame{i}.png', arr=overlay_image)

#########################
#     Sub-Routine       #
########################

def addWeighted(Image1, Image2, alpha, beta, mask):
    height = Image1.shape[0]
    width  = Image1.shape[1]
    sum_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if mask[i][j] == 1:
                sum_image[i][j] = beta*Image2[i][j]
            else:
                sum_image[i][j] = alpha*Image1[i][j]*255
    return sum_image

#---------------Execution---------------#


if __name__ == '__main__':
    main()