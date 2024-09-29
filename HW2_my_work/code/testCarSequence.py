import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# Q1.3

#########################
#     Main-Routine      #
#########################
def main():
    # write your script here, we recommend the above libraries for making your animation

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1e-2,
        help='dp threshold of Lucas-Kanade for terminating optimization',
    )
    args = parser.parse_args()
    num_iters = args.num_iters
    threshold = args.threshold

    seq = np.load("../data/carseq.npy")
    rect = [59, 116, 145, 151]
    rect_mat = np.zeros((int(seq.shape[2]), 4))
    rows = seq.shape[0]
    cols = seq.shape[1]
    p0 = np.zeros(2)

    for i in range(int(seq.shape[2])):
        rect_mat[i] = np.array(rect)
        if i == (int(seq.shape[2])-1):
            break

        p = LucasKanade(seq[:, :, i], seq[:, :, i+1], rect, threshold, num_iters, p0)
        rect[0] += p[0]
        rect[1] += p[1]
        rect[2] += p[0]
        rect[3] += p[1]
        rect[0] = np.clip(rect[0], 0, cols - 1)
        rect[1] = np.clip(rect[1], 0, rows - 1)
        rect[2] = np.clip(rect[2], 0, cols - 1)
        rect[3] = np.clip(rect[3], 0, rows - 1)
        p0 = np.zeros(2)
        tracked = [x for x in rect]

        print(f"i = {i}, rect = {tracked}")
        if i == 0 or i==99 or i==199 or i==299 or i==399:
            fig, ax = plt.subplots()
            ax.imshow(seq[:, :, i+1], cmap='gray')
            width = tracked[2] - tracked[0]
            height = tracked[3] - tracked[1]
            rect_patch = patches.Rectangle((tracked[0], tracked[1]), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect_patch)
            ax.axis('off')
            plt.savefig(f'../carseq_frame{i+1}.png', bbox_inches='tight', pad_inches=0)
            #plt.show()

    np.save('../carseqrects.npy', rect_mat)

#---------------Execution---------------#
if __name__ == '__main__':
    main()