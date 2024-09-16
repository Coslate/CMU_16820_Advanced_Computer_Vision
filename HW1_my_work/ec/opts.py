'''
Hyperparameters wrapped in argparse
This file contains most of tuanable parameters for this homework


You can change the values by changing their default fields or by command-line
arguments. For example, "python q2_1_4.py --sigma 0.15 --ratio 0.7"
'''

import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='16-720 HW2: Homography')

    ## Feature detection (requires tuning)
    parser.add_argument('--sigma', type=float, default=0.15,
                        help='threshold for corner detection using FAST feature detector')
    parser.add_argument('--ratio', type=float, default=0.7,
                        help='ratio for BRIEF feature descriptor')

    ## Ransac (requires tuning)
    parser.add_argument('--max_iters', type=int, default=500,
                        help='the number of iterations to run RANSAC for')
    parser.add_argument('--inlier_tol', type=float, default=2.0,
                        help='the tolerance value for considering a point to be an inlier')

    ## Additional options (add your own hyperparameters here)
    parser.add_argument("--input_src_file", "-in_srcf", help="the file path for 'ar_source.mov'")
    parser.add_argument("--input_dst_file", "-in_dstf", help="the file path for 'book.mov'.")
    parser.add_argument("--output_file", "-out_f", help="the file path for output generated video.")
    parser.add_argument("--down_sample_factor", "-dsf", type=int, default=2, help="the factor of width/hight to downsampling.")
    parser.add_argument("--frame_interval_same", "-fis", type=int, default=2, help="the number of frames that are estimated same Homography due to minor change.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    ##
    opts = parser.parse_args()

    return opts
