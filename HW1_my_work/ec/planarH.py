import numpy as np
import cv2
import random

#to-be-deleted
from opts import get_opts


def computeH(x1, x2):
    #Q2.2.1
    # TODO: Compute the homography between two sets of points
    # Construct A matrix
    A = np.zeros((2*x1.shape[0], 9))
    for i in range(x1.shape[0]):
        xi1 = x1[i][0]
        yi1 = x1[i][1]
        xi2 = x2[i][0]
        yi2 = x2[i][1]
        A[2*i+0] = [xi2, yi2, 1,   0,   0, 0,  -xi1*xi2, -xi1*yi2, -xi1]
        A[2*i+1] = [  0,   0, 0, xi2, yi2, 1,  -yi1*xi2, -yi1*yi2, -yi1]

    # Construct ATA
    ATA = A.T@A

    # Calculate eigenvalues, eigenvectors
    eigenval, eigenvec = np.linalg.eig(ATA)

    # Choose the eigenvector w/ smallest eigenvalues
    smallest_eigenvec = eigenvec[:, np.argsort(eigenval)[0]]
    H2to1 = np.reshape(smallest_eigenvec, (3, 3))
    H2to1 = np.array(H2to1, dtype=np.float32)
    

    #test
    '''
    x2_homo = np.hstack([x2, np.ones((x2.shape[0], 1))]) #Nx3
    x1_recover_all = H2to1@x2_homo.T #3xN
    x1_recover_all = x1_recover_all.T #Nx3
    x1_col1 = x1_recover_all[:, 0]/x1_recover_all[:, 2]
    x1_col2 = x1_recover_all[:, 1]/x1_recover_all[:, 2]
    x1_recover_all = np.column_stack((x1_col1, x1_col2)) #Nx2
    print(f"original, x1_recover_all = {x1_recover_all}")
    '''

    return H2to1

def computeDist(x, cx):
    # x : np.array: (2,)
    # cx: np.array: (2,)
    return np.linalg.norm(x-cx)

def computeLongestDist(x, cx):
    # x : np.array: (N, 2)
    # cx: np.array: (2,)
    '''
    dist = -1
    cand = np.array([0, 0])
    for xi in x:
        computed_dist = computeDist(xi, cx)
        if computed_dist > dist:
            dist = computed_dist
            cand[0] = xi[0]
            cand[1] = xi[1]
    return dist, cand
    '''
    return np.max(np.sqrt(np.sum((x-cx)**2, axis=1)))

def computeH_norm(x1, x2):
    #Q2.2.2
    # TODO: Compute the centroid of the points
    cx1, cy1 = np.mean(x1[:, 0]), np.mean(x1[:, 1])
    cx2, cy2 = np.mean(x2[:, 0]), np.mean(x2[:, 1])
    c1 = np.array([cx1, cy1])
    c2 = np.array([cx2, cy2])

    # TODO: Shift the origin of the points to the centroid
    # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    # Merge these two TODO to form T1 and T2
    # Calculate the largest distance from the center (cx1, cy1), (cx2, cy2)
    longest_dist1 = computeLongestDist(x1, c1)
    longest_dist2 = computeLongestDist(x2, c2)
    norm_factor1 = np.sqrt(2)/longest_dist1
    norm_factor2 = np.sqrt(2)/longest_dist2

    # TODO: Similarity transform 1
    T1 = np.zeros((3, 3))
    T1[0][0] = norm_factor1
    T1[0][2] = -norm_factor1*c1[0]
    T1[1][1] = norm_factor1
    T1[1][2] = -norm_factor1*c1[1]
    T1[2][2] = 1
    x1_homo = np.hstack([x1, np.ones((x1.shape[0], 1))]) #Nx3
    x1_norm = T1@x1_homo.T #3xN
    x1_norm = x1_norm.T #Nx3
    x1_col1 = x1_norm[:, 0]/x1_norm[:, 2]
    x1_col2 = x1_norm[:, 1]/x1_norm[:, 2]
    x1_norm = np.column_stack((x1_col1, x1_col2)) #Nx2

    # TODO: Similarity transform 2
    T2 = np.zeros((3, 3))
    T2[0][0] = norm_factor2
    T2[0][2] = -norm_factor2*c2[0]
    T2[1][1] = norm_factor2
    T2[1][2] = -norm_factor2*c2[1]
    T2[2][2] = 1
    x2_homo = np.hstack([x2, np.ones((x2.shape[0], 1))]) #Nx3
    x2_norm = T2@x2_homo.T #3xN
    x2_norm = x2_norm.T #Nx3
    x2_col1 = x2_norm[:, 0]/x2_norm[:, 2]
    x2_col2 = x2_norm[:, 1]/x2_norm[:, 2]
    x2_norm = np.column_stack((x2_col1, x2_col2)) #Nx2

    # TODO: Compute homography
    H2to1_norm = computeH(x1_norm, x2_norm)

    # TODO: Denormalization
    H2to1 = np.linalg.inv(T1)@H2to1_norm@T2

    return H2to1

def computeH_ransac(locs1, locs2, opts, fit_inlier_last: bool = False, fit_inlier_last_num: int = 4, down_sample_factor: int = 1):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    if locs1.shape[0] < 4:
       raise ValueError(f"Error: The number of match point is {len(locs1)}, smaller than 4, not sufficient to calculate a Homography.")

    max_inlier_num = -1
    max_inliers = np.zeros(locs1.shape[0], dtype = int)
    for i in range(max_iters):
        # Randomly sample 4 points from locs1, locs2
        random_index = random.sample(range(0, locs1.shape[0]), 4)
        locs_sample1 = locs1[random_index, :]
        locs_sample2 = locs2[random_index, :]

        # Calculate the Homography
        H2to1 = computeH_norm(locs_sample1, locs_sample2)

        # Calculate the inline number by recovering x1 points
        locs2_homo = np.hstack([locs2, np.ones((locs2.shape[0], 1))]) #Nx3
        locs1_recover_all = H2to1@locs2_homo.T #3xN
        locs1_recover_all = locs1_recover_all.T #Nx3
        locs1_col1 = locs1_recover_all[:, 0]/locs1_recover_all[:, 2]
        locs1_col2 = locs1_recover_all[:, 1]/locs1_recover_all[:, 2]
        locs1_recover_all = np.column_stack((locs1_col1, locs1_col2)) #Nx2

        inliers = np.sum((locs1_recover_all-locs1)**2, axis=1) <= inlier_tol**2
        '''
        for i in range(locs1_recover_all.shape[0]):
            recovered_pt = locs1_recover_all[i]
            original_pt  = locs1[i]
            dist = computeDist(recovered_pt, original_pt)
            if dist <= inlier_tol:
                inliers[i] = 1
        '''

        inlier_num = np.sum(inliers)
        if inlier_num > max_inlier_num:
            max_inlier_num = inlier_num
            max_inliers    = inliers
            bestH2to1 = H2to1

    inliers = max_inliers

    #Fit the inlier again
    if fit_inlier_last:
        indices = [i for i, x in enumerate(inliers) if x == 1]
        if len(indices) >= fit_inlier_last_num:
            locs1_opt = locs1[indices, :]
            locs2_opt = locs2[indices, :]
            bestH2to1 = computeH_norm(locs1_opt, locs2_opt)

    if down_sample_factor > 1:
        bestH2to1[0, 2] *= down_sample_factor
        bestH2to1[1, 2] *= down_sample_factor

    #test
    '''
    x2 = locs2
    x2_homo = np.hstack([x2, np.ones((x2.shape[0], 1))]) #Nx3
    x1_recover_all = bestH2to1@x2_homo.T #3xN
    x1_recover_all = x1_recover_all.T #Nx3
    x1_col1 = x1_recover_all[:, 0]/x1_recover_all[:, 2]
    x1_col2 = x1_recover_all[:, 1]/x1_recover_all[:, 2]
    x1_recover_all = np.column_stack((x1_col1, x1_col2)) #Nx2
    print(f"ransac, x1_recover_all = {x1_recover_all}")
    '''

    return bestH2to1, inliers

def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    

    # TODO: Create mask of same size as template
    mask = np.full((template.shape[0], template.shape[1], 1), 255, dtype=np.uint8)
    composite_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # TODO: Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))

    # TODO: Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))

    # TODO: Use mask to combine the warped template and the image
    inverse_mask    = cv2.bitwise_not(warped_mask)
    masked_template = cv2.bitwise_and(warped_template, warped_template, mask=warped_mask)
    masked_img      = cv2.bitwise_and(img, img, mask=inverse_mask)
    composite_img   = cv2.add(masked_template, masked_img)
    '''
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if warped_mask[i, j] == 255:
                composite_img[i, j] = warped_template[i, j]
            else:
                composite_img[i, j] = img[i, j]
    '''

    return composite_img


#to-be-deleted
'''
if __name__ == "__main__":
    opts = get_opts()

    x1 = np.array([[30, 28],
                   [22, 24],
                   [45, 10],
                   [78, 45],
                   [23, 25],
                   [13, 22]])

    x2 = np.array([[205, 183],
                   [300, 156],
                   [100, 256],
                   [405, 179],
                   [301, 155],
                   [100, 120]])
    H2to1_norm   = computeH_norm(x1, x2)
    H2to1        = computeH(x1, x2)
    H2to1_ransac, inliers = computeH_ransac(x1, x2, opts)
    H2to1_opencv, status = cv2.findHomography(x2, x1)
    print(f"num_inliers = {np.sum(inliers==1)}")
    print(f"inliers = {inliers}")
    print(f"H2to1_norm = {H2to1_norm}")
    print(f"H2to1 = {H2to1}")
    print(f"H2to1_ransac = {H2to1_ransac}")
    print(f"H2to1_opencv (opencv) = {H2to1_opencv}")

    x2_homo = np.hstack([x2, np.ones((x2.shape[0], 1))]) #Nx3
    x1_recover_all = H2to1_ransac@x2_homo.T #3xN
    x1_recover_all = x1_recover_all.T #Nx3
    x1_col1 = x1_recover_all[:, 0]/x1_recover_all[:, 2]
    x1_col2 = x1_recover_all[:, 1]/x1_recover_all[:, 2]
    x1_recover_all = np.column_stack((x1_col1, x1_col2)) #Nx2
    print(f"ransac, x1_recover_all = {x1_recover_all}")

    x1_recover_all = H2to1@x2_homo.T #3xN
    x1_recover_all = x1_recover_all.T #Nx3
    x1_col1 = x1_recover_all[:, 0]/x1_recover_all[:, 2]
    x1_col2 = x1_recover_all[:, 1]/x1_recover_all[:, 2]
    x1_recover_all = np.column_stack((x1_col1, x1_col2)) #Nx2
    print(f"original, x1_recover_all = {x1_recover_all}")

    x1_recover_all = H2to1_norm@x2_homo.T #3xN
    x1_recover_all = x1_recover_all.T #Nx3
    x1_col1 = x1_recover_all[:, 0]/x1_recover_all[:, 2]
    x1_col2 = x1_recover_all[:, 1]/x1_recover_all[:, 2]
    x1_recover_all = np.column_stack((x1_col1, x1_col2)) #Nx2
    print(f"norm, x1_recover_all = {x1_recover_all}")

    x1_recover_all = H2to1_opencv@x2_homo.T #3xN
    x1_recover_all = x1_recover_all.T #Nx3
    x1_col1 = x1_recover_all[:, 0]/x1_recover_all[:, 2]
    x1_col2 = x1_recover_all[:, 1]/x1_recover_all[:, 2]
    x1_recover_all = np.column_stack((x1_col1, x1_col2)) #Nx2
    print(f"opencv, x1_recover_all = {x1_recover_all}")
'''