import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
# Q1.4

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
    parser.add_argument(
        '--template_threshold',
        type=float,
        default=5,
        help='threshold for determining whether to update template',
    )

    parser.add_argument(
        '--output_folder',
        type=str,
        default='.',
        help='output folder for carseqrects-wcrt.npy and carseq_frame*-wcrt.png',
    )

    args = parser.parse_args()
    num_iters = args.num_iters
    threshold = args.threshold
    template_threshold = args.template_threshold
    output_folder = args.output_folder

    seq = np.load("../data/carseq.npy")
    rect = [59, 116, 145, 151]
    rect_mat = np.zeros((int(seq.shape[2]), 4))
    rows = seq.shape[0]
    cols = seq.shape[1]
    p0  = np.zeros(2)
    p01 = np.zeros(2)
    Itn =0
    It1 =0
    Tn_rect     = [x for x in rect]
    T1_rect     = [x for x in rect]
    T1_rect_upd = [x for x in rect]
    Tn_rect_upd = [x for x in rect]
    tracked     = [x for x in rect]

    for i in range(int(seq.shape[2])):
        rect_mat[i] = np.array(tracked)
        if i == (int(seq.shape[2])-1):
            break

        pn = LucasKanade(seq[:, :, Itn],  seq[:, :, i+1], Tn_rect, threshold, num_iters, p0,  print_t1=False)
        Tn_rect_upd[0] = Tn_rect[0] + pn[0]
        Tn_rect_upd[1] = Tn_rect[1] + pn[1]
        Tn_rect_upd[2] = Tn_rect[2] + pn[0]
        Tn_rect_upd[3] = Tn_rect[3] + pn[1]
        Tn_rect_upd[0] = np.clip(Tn_rect_upd[0], 0, cols - 1)
        Tn_rect_upd[1] = np.clip(Tn_rect_upd[1], 0, rows - 1)
        Tn_rect_upd[2] = np.clip(Tn_rect_upd[2], 0, cols - 1)
        Tn_rect_upd[3] = np.clip(Tn_rect_upd[3], 0, rows - 1)

        p01[0] = Tn_rect_upd[0] - T1_rect[0]
        p01[1] = Tn_rect_upd[1] - T1_rect[1]
        p1 = LucasKanade(seq[:, :, It1], seq[:, :, i+1], T1_rect, threshold, num_iters, p01, print_t1=False)
        T1_rect_upd[0] = T1_rect[0] + p1[0]
        T1_rect_upd[1] = T1_rect[1] + p1[1]
        T1_rect_upd[2] = T1_rect[2] + p1[0]
        T1_rect_upd[3] = T1_rect[3] + p1[1]
        T1_rect_upd[0] = np.clip(T1_rect_upd[0], 0, cols - 1)
        T1_rect_upd[1] = np.clip(T1_rect_upd[1], 0, rows - 1)
        T1_rect_upd[2] = np.clip(T1_rect_upd[2], 0, cols - 1)
        T1_rect_upd[3] = np.clip(T1_rect_upd[3], 0, rows - 1)

        if i == 0:
            # Store T1
            Itn = 1
            Tn_rect = [x for x in Tn_rect_upd]
            p0 = np.zeros(2)

            # Store T1
            It1 = 1
            T1_rect = [x for x in T1_rect_upd]
            p01     = np.zeros(2)

            tracked = [x for x in Tn_rect]
        else:
            # Drift Correction using T1
            p1_arr = np.array([T1_rect[0]+p1[0], T1_rect[1]+p1[1]])
            pn_arr = np.array([Tn_rect[0]+pn[0], Tn_rect[1]+pn[1]])
            if np.linalg.norm((pn_arr-p1_arr), ord=2) < template_threshold:
                print(f"i = {i}, tn")
                Itn = i+1
                Tn_rect = [x for x in T1_rect_upd]
                p0[0] = T1_rect_upd[0] - Tn_rect[0]
                p0[1] = T1_rect_upd[1] - Tn_rect[1]

                tracked = [x for x in Tn_rect_upd]
            else:
                print(f"i = {i}, t1")
                tracked = [x for x in Tn_rect_upd]
                p0[0] = T1_rect_upd[0] - Tn_rect[0]
                p0[1] = T1_rect_upd[1] - Tn_rect[1]

        print(f"i = {i}, tracked = {tracked}")
        if i == 0 or i==99 or i==199 or i==299 or i==399:
            fig, ax = plt.subplots()
            ax.imshow(seq[:, :, i+1], cmap='gray')
            width = tracked[2] - tracked[0]
            height = tracked[3] - tracked[1]
            rect_patch = patches.Rectangle((tracked[0], tracked[1]), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect_patch)
            ax.axis('off')
            plt.savefig(f'{output_folder}/carseq_frame{i+1}-wcrt.png', bbox_inches='tight', pad_inches=0)
            #plt.show()

    np.save(f'{output_folder}/carseqrects-wcrt.npy', rect_mat)

#---------------Execution---------------#
if __name__ == '__main__':
    main()