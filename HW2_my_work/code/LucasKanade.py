import numpy as np
from scipy.interpolate import RectBivariateSpline
import skimage.color
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


EPSILON = pow(10, -4)

def WarpImage(p, rect, rows, cols):
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]

    # Construct the grid
    x_range = np.linspace(x1, x2, round(x2-x1+1))
    y_range = np.linspace(y1, y2, round(y2-y1+1))
    grid_x, grid_y = np.meshgrid(x_range, y_range, indexing='ij')

    # Define the warp matrix: Translation
    p_x = float(p[0])
    p_y = float(p[1])
    warp_matrix = np.array([[1, 0, p_x],
                            [0, 1, p_y],
                            [0, 0, 1  ]], dtype=np.float32)

    # Get the original homogeneous coordinates in rect
    orig_homo_coord = np.vstack([grid_x.ravel(), grid_y.ravel(), np.ones((1, grid_x.shape[0]*grid_x.shape[1]))])
    orig_x = orig_homo_coord[0, :]
    orig_y = orig_homo_coord[1, :]

    # Perform the warping
    new_homo_coord = warp_matrix@orig_homo_coord

    # Interpolate
    new_x = new_homo_coord[0, :]/new_homo_coord[2, :]
    new_y = new_homo_coord[1, :]/new_homo_coord[2, :]

    new_x = np.clip(new_x, 0, cols - 1)
    new_y = np.clip(new_y, 0, rows - 1)    
    return (new_x, new_y), (orig_x, orig_y)

def calculate_inverse(H, lam_epsilon=EPSILON):
    try:
        H_inverse = np.linalg.inv(H)
        return H_inverse
    except np.linalg.LinAlgError:
        H_lmreg = H+lam_epsilon*np.eye(H.shape[0])
        H_inverse = np.linalg.inv(H_lmreg)
        return H_inverse


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2), print_t1=False):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    # Initialize parameters
    p = p0.copy()
    rows = It.shape[0]
    cols = It.shape[1]
    new_rect = [x for x in rect]

    # Create spline for interpolation
    it_spline  = RectBivariateSpline(np.arange(0, rows), np.arange(0, cols), It)
    it1_spline = RectBivariateSpline(np.arange(0, rows), np.arange(0, cols), It1)
    #blurred_image_3x3 = cv2.GaussianBlur(It1, (5, 5), 0)
    #sobel_x = cv2.Sobel(It1, cv2.CV_64F, 1, 0, ksize=3)
    #sobel_y = cv2.Sobel(It1, cv2.CV_64F, 0, 1, ksize=3)
    #sobel_x_spline = RectBivariateSpline(np.arange(0, rows), np.arange(0, cols), sobel_x)
    #sobel_y_spline = RectBivariateSpline(np.arange(0, rows), np.arange(0, cols), sobel_y)
    scharr_x = cv2.Scharr(It1, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(It1, cv2.CV_64F, 0, 1)
    scharr_x_spline = RectBivariateSpline(np.arange(0, rows), np.arange(0, cols), scharr_x)
    scharr_y_spline = RectBivariateSpline(np.arange(0, rows), np.arange(0, cols), scharr_y)

    if print_t1:
        fig, ax = plt.subplots()
        ax.set_title("T1")
        ax.imshow(It, cmap='gray')
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        rect_patch = patches.Rectangle((rect[0], rect[1]), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect_patch)
        ax.axis('off')
        plt.show()


    for i in range(int(num_iters)):
        #Warp the image
        (warped_x, warped_y), (orig_x, orig_y) = WarpImage(p, new_rect, rows, cols)

        #Form Matrix A
        #sobel_x_roi = np.reshape(sobel_x_spline.ev(warped_y, warped_x), (1, -1))
        #sobel_y_roi = np.reshape(sobel_y_spline.ev(warped_y, warped_x), (1, -1))
        #A = np.hstack([sobel_x_roi.T, sobel_y_roi.T])
        scharr_x_roi = np.reshape(scharr_x_spline.ev(warped_y, warped_x), (1, -1))
        scharr_y_roi = np.reshape(scharr_y_spline.ev(warped_y, warped_x), (1, -1))
        A = np.hstack([scharr_x_roi.T, scharr_y_roi.T])
        #print(f"A.shape = {A.shape}")

        #Form Matrix b
        it_roi  = np.reshape(it_spline.ev(orig_y, orig_x), (-1, 1))
        it1_roi = np.reshape(it1_spline.ev(warped_y, warped_x), (-1, 1))
        #b = cv2.subtract(it_roi, it1_roi)
        b = (it_roi - it1_roi).reshape(-1, 1)
        #print(f"b.shape = {b.shape}")

        #Calculate Approximate Hessian
        H = A.T@A
        H_inverse = calculate_inverse(H, lam_epsilon=EPSILON)

        #Calculate delta_p
        delta_p = H_inverse@(A.T@b)
        delta_p = np.ravel(delta_p)
        p = p + delta_p

        #x1 = np.clip(rect[0] + p[0], 0, cols-1)
        #y1 = np.clip(rect[1] + p[1], 0, rows-1)
        #x2 = np.clip(rect[2] + p[0], 0, cols-1)
        #y2 = np.clip(rect[3] + p[1], 0, rows-1)
        #new_rect = [x1, y1, x2, y2]

        if np.linalg.norm(delta_p, ord=2)**2 < threshold:
            break

        '''
        image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        image = It.copy()
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), 255, 2)
        cv2.rectangle(image, (rect[0]+delta_p[0], rect[1]+delta_p[1]), (rect[2]+delta_p[0], rect[3]+delta_p[1]), 0, 2)
        cv2.imshow("image", image)
        cv2.imshow("gray_It", It)
        cv2.imshow("warped_It", warped_It)
        cv2.waitKey(0)
        '''

    return p
