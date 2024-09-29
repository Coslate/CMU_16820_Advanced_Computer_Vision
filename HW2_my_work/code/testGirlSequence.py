import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
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

    seq = np.load("../data/girlseq.npy")
    rect = [280, 152, 330, 318]
    rect_mat = np.zeros((int(seq.shape[2]), 4))
    rows = seq.shape[0]
    cols = seq.shape[1]
    p0 = np.zeros(2)

    for i in range(int(seq.shape[2])):
        rect_mat[i] = np.array(rect)
        if i == (int(seq.shape[2])-1):
            break

        p = LucasKanade(seq[:, :, i], seq[:, :, i+1], rect, threshold, num_iters, p0)
        p0 = p.copy()
        tracked = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
        print(f"i = {i}, rect = {tracked}")
        if i == 0 or i==19 or i==39 or i==59 or i==79:
            fig, ax = plt.subplots()
            ax.imshow(seq[:, :, i+1], cmap='gray')
            width = tracked[2] - tracked[0]
            height = tracked[3] - tracked[1]
            rect_patch = patches.Rectangle((tracked[0], tracked[1]), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect_patch)
            ax.axis('off')
            plt.savefig(f'../girlseq_frame{i+1}.png', bbox_inches='tight', pad_inches=0)
            #plt.show()

    np.save('../girlseqrects.npy', rect_mat)

#---------------Execution---------------#
if __name__ == '__main__':
    main()