import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

EPSILON = pow(10, -4)

def param2Matrix(p):
    M = np.array([[1+p[0], p[1]  , p[2]],
                  [p[3]  , 1+p[4], p[5]],], dtype=np.float32)

    return M

def param2MatrixHomo(p):
    M = np.array([[1+p[0], p[1]  , p[2]],
                  [p[3]  , 1+p[4], p[5]],
                  [0     , 0     , 1   ]], dtype=np.float32)

    return M

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
    warp_matrix = param2MatrixHomo(p)

    # Get the original homogeneous coordinates in rect
    orig_homo_coord = np.vstack([grid_x.ravel(), grid_y.ravel(), np.ones((1, grid_x.shape[0]*grid_x.shape[1]))])
    orig_x = orig_homo_coord[0, :]
    orig_y = orig_homo_coord[1, :]

    # Perform the warping
    new_homo_coord = warp_matrix@orig_homo_coord

    # Homogeneous to pixel
    new_x = new_homo_coord[0, :]/new_homo_coord[2, :]
    new_y = new_homo_coord[1, :]/new_homo_coord[2, :]

    # Remove out-of-range pixels
    valid_idx = (new_x >= 0) & (new_x < cols) & (new_y >= 0) & (new_y < rows)
    valid_warped_x = new_x[valid_idx]
    valid_warped_y = new_y[valid_idx]
    valid_orig_x   = orig_x[valid_idx]
    valid_orig_y   = orig_y[valid_idx]

    return (valid_warped_x, valid_warped_y), (valid_orig_x, valid_orig_y)

def calculate_inverse(H, lam_epsilon=EPSILON):
    try:
        H_inverse = np.linalg.inv(H)
        return H_inverse
    except np.linalg.LinAlgError:
        H_lmreg = H+lam_epsilon*np.eye(H.shape[0])
        H_inverse = np.linalg.inv(H_lmreg)
        return H_inverse

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ################### TODO Implement Lucas Kanade Affine ###################

    p = np.zeros(6)
    rows, cols= It.shape[0], It.shape[1]
    targeted_rows, targeted_cols= It1.shape[0], It1.shape[1]
    new_rect = [0, 0, cols-1, rows-1]

    # Create spline for interpolation
    it_spline  = RectBivariateSpline(np.arange(0, rows), np.arange(0, cols), It)
    it1_spline = RectBivariateSpline(np.arange(0, targeted_rows), np.arange(0, targeted_cols), It1)
    #blurred_image_3x3 = cv2.GaussianBlur(It1, (5, 5), 0)
    #sobel_x = cv2.Sobel(It1, cv2.CV_64F, 1, 0, ksize=3)
    #sobel_y = cv2.Sobel(It1, cv2.CV_64F, 0, 1, ksize=3)
    #sobel_x_spline = RectBivariateSpline(np.arange(0, rows), np.arange(0, cols), sobel_x)
    #sobel_y_spline = RectBivariateSpline(np.arange(0, rows), np.arange(0, cols), sobel_y)
    scharr_x = cv2.Scharr(It1, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(It1, cv2.CV_64F, 0, 1)
    scharr_x_spline = RectBivariateSpline(np.arange(0, targeted_rows), np.arange(0, targeted_cols), scharr_x)
    scharr_y_spline = RectBivariateSpline(np.arange(0, targeted_rows), np.arange(0, targeted_cols), scharr_y)

    for iter in range(int(num_iters)):
        #Warp the image
        (warped_x, warped_y), (orig_x, orig_y) = WarpImage(p, new_rect, targeted_rows, targeted_cols)

        #Form Matrix A
        #sobel_x_roi = np.reshape(sobel_x_spline.ev(warped_y, warped_x), (1, -1))
        #sobel_y_roi = np.reshape(sobel_y_spline.ev(warped_y, warped_x), (1, -1))
        #A = np.hstack([sobel_x_roi.T, sobel_y_roi.T])
        A = np.zeros((warped_x.shape[0], 6)) #Nx6
        scharr_x_roi = np.reshape(scharr_x_spline.ev(warped_y, warped_x), (-1, 1)) #Nx1
        scharr_y_roi = np.reshape(scharr_y_spline.ev(warped_y, warped_x), (-1, 1)) #Nx1
        for i in range(int(warped_x.shape[0])):
            grad_mat = np.array([scharr_x_roi[i], scharr_y_roi[i]]).reshape(1, 2)
            jacb_mat = np.array([[orig_x[i], orig_y[i], 1, 0        , 0        , 0],
                                 [0        , 0        , 0, orig_x[i], orig_y[i], 1]])
            A[i] = grad_mat@jacb_mat
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
        #print(f"H_inverse.shape = {H_inverse.shape}")

        #Calculate delta_p
        delta_p = H_inverse@(A.T@b)
        delta_p = np.ravel(delta_p)
        p = p + delta_p

        err = np.linalg.norm(delta_p, ord=2)**2
        if iter%500 == 0:
            print(f"iter = {iter}, err = {err}")
        if err < threshold:
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

    M = param2Matrix(p)
    return M

#---------------Execution---------------#
if __name__ == '__main__':
    num_iters = 1e4
    #threshold = 1e-11
    threshold = 1e-8
    seq = np.load("../data/aerialseq.npy")
    template = seq[:, :, 0]
    It1 = seq[:, :, 1]
    M = np.array([[1.1, 0.5  , 10], 
                  [0.1, 0.8  , 18]], dtype=np.float32)

    #linear_part_M = M[:, :2]
    #translation_M = -M[:, 2]
    #linear_part_inv = np.linalg.inv(linear_part_M)
    #warped_image_aff = affine_transform(template.T, linear_part_inv, offset=translation_M, order=1).T
    #print(f"template.shape = {template.shape}")

    # Get the image dimensions (height and width)
    height, width = template.shape[:2]
    print(f"height = {height}, width = {width}")

    # Apply the affine transformation
    warped_image = cv2.warpAffine(template, M, (width, height), flags=cv2.INTER_LINEAR)

    #cv2.imshow('original_image', template)
    #cv2.imshow('warped_image_cv', warped_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cal_M = LucasKanadeAffine(template, warped_image, threshold, num_iters)
    print(f"M = {M}")
    print(f"cal_M = {cal_M}")