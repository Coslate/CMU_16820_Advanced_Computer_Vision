## HW1 - Planar Homography

#### Instructions
For Q3.1 Please first go to python/ folder and use the following command line to reproduce the result stored in ../result/:
python ./ar.py --input_src_file ../data/ar_source.mov  --input_dst_file ../data/book.mov  --output_file ../result/ar.avi --sigma 0.08 --ratio 0.61 --max_iters 1000 --inlier_tol 1

For Q3.2 to run the profiling of the current program (ar.py), run the following command, and check the result in ../result_profiling/:
python ./ar_profiling.py --input_src_file ../data/ar_source.mov  --input_dst_file ../data/book.mov  --output_file ../result_profiling/ar.avi --sigma 0.08 --ratio 0.61 --max_iters 40 --inlier_tol 3

For Q4 to run the panoroma, run the following command, and check the result in ../result_panorama/, please be noted to specify --input_left_img_file and --input_right_img_file for the left and righ images to produce the panorama, 
and please remember to specify the output file name with --output_file:
python ./panaroma.py --input_left_img_file ../data/my_pano_left.jpg --input_right_img_file ../data/my_pano_right.jpg --max_iters 1000 --inlier_tol 3 --output_file ../result_panorama/Q4_Panorama_Result.png