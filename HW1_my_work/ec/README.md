## HW1 - Planar Homography

#### Instructions
For Q3.2 Please first go to ec/ folder and use the following command line to reproduce the result stored in ../result_ec/:
python ./ar_ec.py --input_src_file ../data/ar_source.mov  --input_dst_file ../data/book.mov  --output_file ../result_ec/ar_ec.avi --sigma 0.08 --ratio 0.61 --max_iters 40 --inlier_tol 3 --down_sample_factor 2 --frame_interval_same  1

The ../result_ec is specified by --output_file argument, the folder contains the following output files:
1. ../result_ec/ar_ec.avi, the resulted video that is written out by the program.
2. ../result_ec/ar_ec_profiling.txt, the file contains the calculated FPS.

