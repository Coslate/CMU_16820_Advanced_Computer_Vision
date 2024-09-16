import numpy as np
import cv2

# Import necessary functions

from loadVid import loadVid


# Q3.2

#Import necessary functions
from opts import get_opts
from typing import List, Any, Tuple
from displayMatch import displayMatched
from planarH import compositeH
from matchPics import matchPics
from planarH   import computeH_ransac
import os
import time
import shutil

#Write script for Q3.1

#########################
#     Main-Routine      #
#########################
def main():
    # Input arguments/Load input videos
    opts = get_opts()
    out_folder = os.path.dirname(opts.output_file)
    if not os.path.exists(out_folder): 
        os.makedirs(out_folder)
    else: 
        shutil.rmtree(out_folder)
        os.makedirs(out_folder)

    frames_ar_source = loadVid(opts.input_src_file)
    frames_book      = loadVid(opts.input_dst_file)

    # Process each frame
    frame_result = generalHarryPoterize(frames_ar_source, frames_book, opts, out_folder)

    # Wriet out video
    writeOutVideo(frame_result, opts)

    # Wriet out three distinct timestamps
    #frameidx_book_left  = 123
    #frameidx_book_cnter = 230
    #frameidx_book_right = 331
    #cv2.imwrite(f"{out_folder}/Q3_1_ar_result_overlay_left.jpg", frame_result[frameidx_book_left])
    #cv2.imwrite(f"{out_folder}/Q3_1_ar_result_overlay_right.jpg", frame_result[frameidx_book_right])
    #cv2.imwrite(f"{out_folder}/Q3_1_ar_result_overlay_cnter.jpg", frame_result[frameidx_book_cnter])

#########################
#     Sub-Routine       #
#########################
def writeOutVideo(frame_result: List[Any], opts) -> None:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (frame_result[0].shape[1], frame_result[0].shape[0])
    out = cv2.VideoWriter(opts.output_file, fourcc, 30.0, frame_size)

    for frame in frame_result:
        out.write(frame)

def generalHarryPoterize(frame_ar_source: List[Any], frames_book: List[Any], opts, out_folder: str) -> List[Any]:
    image_cv_cover      = cv2.imread('../data/cv_cover.jpg')
    min_len             = min(len(frame_ar_source), len(frames_book))
    frame_result        = []
    down_sample_factor    = opts.down_sample_factor
    frame_interval_same_H = opts.frame_interval_same
    total_time         = 0

    for i in range(min_len):
        time_loop_start = time.perf_counter()
        # Start of the Computation
        # Compute Homography
        if i%frame_interval_same_H == 0:
            matches, locs1, locs2 = matchPics(frames_book[i], image_cv_cover, opts, down_sample_factor)
            x1 = locs1[matches[:, 0], :]
            x2 = locs2[matches[:, 1], :]
            x1_correct_pt = np.flip(x1, axis=1)
            x2_correct_pt = np.flip(x2, axis=1)

            try:
                H2to1_ransac, inliers = computeH_ransac(x1_correct_pt, x2_correct_pt, opts, True, 10, down_sample_factor)
            except ValueError:
                H2to1_ransac = last_H
        else:
            H2to1_ransac = last_H

        # Resize the frame of ar_source to the size of frame of book
        image_cover_height, image_cover_width = image_cv_cover.shape[0], image_cv_cover.shape[1]
        ar_center = (int(frame_ar_source[i].shape[1]/2), int(frame_ar_source[i].shape[0]/2)) #(x, y)
        roi_range = (280, 280) 

        image_ar_cover = frame_ar_source[i][max(ar_center[1]-int(roi_range[1]/2), 48) : min(ar_center[1]+int(roi_range[1]/2), 310), max(ar_center[0]-int(roi_range[0]/2), 0) : min(ar_center[0]+int(roi_range[0]/2), 640)]
        resize_image_ar_cover = cv2.resize(image_ar_cover, (image_cv_cover.shape[1], image_cv_cover.shape[0]))

        # Warp & Composite resize_image_ar_cover to frames_book
        composite_img = compositeH(H2to1_ransac, resize_image_ar_cover, frames_book[i])

        # End of the computation
        time_loop_end = time.perf_counter()
        total_time += (time_loop_end-time_loop_start)


        # Add to result
        frame_result.append(composite_img)
        last_H = H2to1_ransac

        cv2.imshow("composite_img", composite_img)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    result_fps = f"Total Speed = 1/(total_time/total_frames) = {1/(total_time/min_len)} FPS"
    print(result_fps)
    with open(f"{out_folder}/ar_ec_profiling.txt", "w") as f:
        f.write(result_fps)
    return frame_result

#---------------Execution---------------#
if __name__ == '__main__':
    main()