import numpy as np
import cv2

#Import necessary functions
from helper import loadVid
from opts import get_opts
from typing import List, Any, Tuple
from displayMatch import displayMatched
from planarH import compositeH
from matchPics import matchPics
from planarH   import computeH_ransac
import os

#Write script for Q3.1

#########################
#     Main-Routine      #
#########################
def main():
    # Input arguments/Load input videos
    opts = get_opts()
    out_folder = os.path.dirname(opts.output_file)
    if not os.path.exists(out_folder): os.makedirs(out_folder)
    frames_ar_source = loadVid(opts.input_src_file)
    frames_book      = loadVid(opts.input_dst_file)

    # Process each frame
    frame_result = generalHarryPoterize(frames_ar_source, frames_book, opts)

    # Wriet out video
    writeOutVideo(frame_result, opts)

    # Wriet out three distinct timestamps
    frameidx_book_left  = 123
    frameidx_book_cnter = 230
    frameidx_book_right = 331
    cv2.imwrite(f"{out_folder}/Q3_1_ar_result_overlay_left.jpg", frame_result[frameidx_book_left])
    cv2.imwrite(f"{out_folder}/Q3_1_ar_result_overlay_right.jpg", frame_result[frameidx_book_right])
    cv2.imwrite(f"{out_folder}/Q3_1_ar_result_overlay_cnter.jpg", frame_result[frameidx_book_cnter])

#########################
#     Sub-Routine       #
#########################
def writeOutVideo(frame_result: List[Any], opts) -> None:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (frame_result[0].shape[1], frame_result[0].shape[0])
    out = cv2.VideoWriter(opts.output_file, fourcc, 30.0, frame_size)

    for frame in frame_result:
        out.write(frame)

def generalHarryPoterize(frame_ar_source: List[Any], frames_book: List[Any], opts) -> List[Any]:
    image_cv_cover      = cv2.imread('../data/cv_cover.jpg')
    min_len             = min(len(frame_ar_source), len(frames_book))
    frame_result        = []

    for i in range(min_len):
        # Compute Homography
        matches, locs1, locs2 = matchPics(frames_book[i], image_cv_cover, opts)
        x1 = locs1[matches[:, 0], :]
        x2 = locs2[matches[:, 1], :]
        x1_correct_pt = np.flip(x1, axis=1)
        x2_correct_pt = np.flip(x2, axis=1)
        try:
            H2to1_ransac, inliers = computeH_ransac(x1_correct_pt, x2_correct_pt, opts)
        except ValueError:
            H2to1_ransac = last_H
        #displayMatched(opts, frames_book[i], image_cv_cover)

        # Resize the frame of ar_source to the size of frame of book
        image_cover_height, image_cover_width = image_cv_cover.shape[0], image_cv_cover.shape[1]
        ar_center = (int(frame_ar_source[i].shape[1]/2), int(frame_ar_source[i].shape[0]/2)) #(x, y)
        roi_range = (280, 280) 

        image_ar_cover = frame_ar_source[i][max(ar_center[1]-int(roi_range[1]/2), 48) : min(ar_center[1]+int(roi_range[1]/2), 310), max(ar_center[0]-int(roi_range[0]/2), 0) : min(ar_center[0]+int(roi_range[0]/2), 640)]
        resize_image_ar_cover = cv2.resize(image_ar_cover, (image_cv_cover.shape[1], image_cv_cover.shape[0]))

        # Warp & Composite resize_image_ar_cover to frames_book
        composite_img = compositeH(H2to1_ransac, resize_image_ar_cover, frames_book[i])
        last_H = H2to1_ransac

        # Add to result
        frame_result.append(composite_img)

        if i%20 == 0: print(f"i = {i}, progress = {(i+1)/min_len*100}%")
        #cv2.imshow("composite_img", composite_img)
        #cv2.waitKey(3000)
        #cv2.destroyAllWindows()

        # Stop if 'q' is pressed
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        '''
        cv2.imshow("image_ar_cover", image_ar_cover)
        cv2.imshow("resize_image_ar_cover", resize_image_ar_cover)
        cv2.imshow("image_cv_cover", image_cv_cover)
        cv2.imshow("composite_img", composite_img)
        cv2.imshow("frame_ar_source", frame_ar_source[i])
        cv2.imshow("frame_book", frames_book[i])
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)
        '''

    return frame_result

#---------------Execution---------------#
if __name__ == '__main__':
    main()