import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from helper import displayEpipolarF, calc_epi_error, toHomogenous, camera2
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2, triangulate

import scipy

# Insert your package here
from q3_1_essential_matrix import essentialMatrix


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def on_key(event):
    """Handle key press events."""
    if event.key == 'escape':  # ESC key
        print("ESC pressed! Exiting...")
        plt.close('all')  # Close the plot
        return

def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")

    # Connect the key press event to the function on_key
    fig.canvas.mpl_connect('key_press_event', on_key)

    while plt.fignum_exists(fig.number):
        clicked_points = plt.ginput(1, mouse_stop=2)

        if len(clicked_points) > 0:
            x, y = clicked_points[0]
            print(f"Clicked at: {x}, {y}")
        plt.draw()
    plt.close(fig)
    print("Exited the loop and program closed.")
    return


"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
"""
def getFSevenPoint(pts1_choice, pts2_choice, pts1_homo, pts2_homo, M, tol, file="debug.log"):
    best_inlier = np.zeros((pts1_homo.shape[0], 1), dtype=bool)
    best_F = None
    best_inlier_cnt = -1

    Fs = sevenpoint(pts1_choice, pts2_choice, M)
    for F in Fs:
        inlier = np.zeros((pts1_homo.shape[0], 1), dtype=bool)
        error_all = np.abs(calc_epi_error(pts1_homo, pts2_homo, F))
        inlier[error_all < tol] = True
        inlier_cnt = np.sum(inlier)

        if inlier_cnt > best_inlier_cnt:
            best_inlier_cnt = inlier_cnt
            best_F = F.copy()
            best_inlier = inlier.copy()

    return best_F, best_inlier, best_inlier_cnt

def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    # TODO: Replace pass by your implementation
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    best_inlier = np.zeros((pts1_homo.shape[0], 1), dtype=bool)
    best_F = None
    best_inlier_cnt = -1

    file = open("debug.log", "w")
    for i in range(nIters):
        inlier = np.zeros((pts1.shape[0], 1), dtype=bool)
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        cur_F, cur_inlier, cur_inlier_cnt = getFSevenPoint(pts1_choice, pts2_choice, pts1_homo, pts2_homo, M, tol, file=file)

        if cur_inlier_cnt > best_inlier_cnt:
            best_inlier_cnt = cur_inlier_cnt
            best_F = cur_F.copy()
            best_inlier = cur_inlier.copy()
    
    return best_F, best_inlier

"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""


def rodrigues(r, epsilon=1e-20):
    # TODO: Replace pass by your implementation
    r_reshape = r.reshape((-1, 1))
    theta = np.linalg.norm(r_reshape) #floating point value
    if theta < epsilon: # == 0
        return np.eye(3)

    u = r_reshape/theta
    u_cross = np.array([[ 0      , -u[2, 0],  u[1, 0]],
                        [ u[2, 0],  0      , -u[0, 0]],
                        [-u[1, 0],  u[0, 0],  0      ]])
    R = np.eye(3)*np.cos(theta) + (1-np.cos(theta))*(u@u.T) + u_cross*np.sin(theta)
    return R

"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""

def sOneHalf(r, epsilon=1e-16):
    if np.linalg.norm(r) == np.pi and ((r[0, 0] < epsilon and r[1, 0] < epsilon and r[2, 0] < 0) or (r[0, 0] < epsilon and r[1, 0] < 0) or (r[0, 0] < 0)):
        return -1*r
    else:
        return r

def invRodrigues(R, epsilon=1e-16, reshape2flat=True):
    # TODO: Replace pass by your implementation
    A = (R - R.T)/2
    lo = np.array([[A[2, 1], A[0, 2], A[1, 0]]]).T
    s = np.linalg.norm(lo)
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1)/2

    if s < epsilon and (c-1) < epsilon: # s == 0 and c == 1
        return np.zeros((3, 1), dtype=np.float32).reshape(-1) if reshape2flat else np.zeros((3, 1), dtype=np.float32)
    elif s < epsilon and (c+1) < epsilon: # s == 0 and c == -1
        # Find non-zero column of R+I matrix
        matrix = R + np.eye(3)
        for i in range(matrix.shape[1]):
            column = matrix[:, i]
            if np.any(column):  # Check if any element in the column is non-zero
                v = column
                break
        # Get the r
        u = v/np.linalg.norm(v)
        return sOneHalf(u*np.pi).reshape(-1) if reshape2flat else sOneHalf(u*np.pi)
    elif s >= epsilon: # s != 0
        u = lo/s
        theta = np.arctan2(s, c)
        return (u*theta).reshape(-1) if reshape2flat else u*theta
    
    raise ValueError("Undefined combinational values of s and c.")

"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # TODO: Replace pass by your implementation
    P = x[:-6].reshape(-1, 3) #Nx3
    r2 = x[-6:-3].reshape(-1, 1) #3x1
    t2 = x[-3:].reshape(-1, 1) #3x1

    # Reconstruct C1
    C1 = K1@M1

    # Reconstruct C2
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))
    C2 = K2@M2

    # Calculate projected points on C1 and C2
    P_homo = np.hstack((P, np.ones((P.shape[0], 1)))) #Nx4
    p1_proj_homo = C1@P_homo.T #3xN 
    p2_proj_homo = C2@P_homo.T #3xN

    # Convert Homogeneous back to normal coordinates
    p1_hat_x = p1_proj_homo.T[:, 0]/p1_proj_homo.T[:, 2]
    p1_hat_y = p1_proj_homo.T[:, 1]/p1_proj_homo.T[:, 2]
    p1_hat   = np.column_stack((p1_hat_x, p1_hat_y)) #Nx2
    p2_hat_x = p2_proj_homo.T[:, 0]/p2_proj_homo.T[:, 2]
    p2_hat_y = p2_proj_homo.T[:, 1]/p2_proj_homo.T[:, 2]
    p2_hat   = np.column_stack((p2_hat_x, p2_hat_y)) #Nx2

    # Calculate the residual 4Nx1 vector
    residual = np.concatenate([(p1-p1_hat).reshape(-1), (p2-p2_hat).reshape(-1)])

    return residual

"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    # Extract r2, t2 from M2_init
    R2 = M2_init[:, :3] # 3x3
    r2 = invRodrigues(R2, 1e-32).reshape(-1, 1) #3x1
    t2 = M2_init[:, 3].reshape(-1, 1) #3x1
    init_params = np.concatenate([P_init.flatten(), r2.flatten(), t2.flatten()])

    # Calculate the starting object loss
    obj_start = np.sum(rodriguesResidual(K1, M1, p1, K2, p2, init_params) ** 2)

    # Optimization
    object_func = lambda params: np.sum(rodriguesResidual(K1, M1, p1, K2, p2, params) ** 2)
    result = scipy.optimize.minimize(
        fun=object_func,         # Objective function (residual function)
        x0=init_params                # Initial guess for parameters
        #method='L-BFGS-B'              # Optimization method (you can try different methods)
    )
    optimized_params = result.x

    # Extract optimized r2, t2, P, and M2
    t2_optimized = optimized_params[-3:].reshape(-1, 1) #3x1
    r2_optimized = optimized_params[-6:-3].reshape(-1, 1) #3x1
    P_optimized  = optimized_params[:-6].reshape(-1, 3) #Nx3
    R2_optimized = rodrigues(r2_optimized)
    M2           = np.hstack((R2_optimized, t2_optimized))

    # Calculate the resulting object loss
    obj_end = np.sum(rodriguesResidual(K1, M1, p1, K2, p2, optimized_params) ** 2)

    return M2, P_optimized, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    noisy_pts1, noisy_pts2 = some_corresp_noisy["pts1"], some_corresp_noisy["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=1000, tol=2)
    print(f"number of inliers = {np.sum(inliers)}\ntotal number of points = {noisy_pts1.shape[0]}\ninlier rate = {np.sum(inliers)/noisy_pts1.shape[0]*100}%")

    #displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(
        noisy_pts2
    )

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2

    '''
    F_eight = eightpoint(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    displayEpipolarF(im1, im2, F_eight)
    '''

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    # Visualization:
    np.random.seed(1)
    correspondence = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading noisy correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    M = np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    """
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    """
    #Step1: Call the ransacF function
    F, inliers = ransacF(pts1, pts2, M=np.max([*im1.shape, *im2.shape]), nIters=1000, tol=2)
    print(f"number of inliers = {np.sum(inliers)}\ntotal number of points = {pts1.shape[0]}\ninlier rate = {np.sum(inliers)/pts1.shape[0]*100}%")

    #displayEpipolarF(im1, im2, F)
    np.savez('q5_3.npz', F, inliers)
    #F = np.load('q5_3.npz')['arr_0']
    #inliers = np.load('q5_3.npz')['arr_1']
    #displayEpipolarF(im1, im2, F)
    #print(f"F = {F}")

    #Step2: Call the findM2 function to find the M (extrinsics) of the second camera
    p1, p2 = pts1[inliers.flatten()], pts2[inliers.flatten()]
    M2_init, C2_init, P_init = findM2(F, p1, p2, intrinsics, filename="q5_3_findM2.npz")
    M1 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)  #[I|0]

    #Step3: Call the bundleAdjustment function to optimize the extrinsics and 3D points
    M2_opt, P_opt, object_start, object_end = bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init)
    print(f"Reprojection error with initial M2 and w = {object_start}")
    print(f"Optimized Reprojection error with optimized M2 and w = {object_end}")

    #Step4: Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    fig = plt.figure()
    ax = Axes3D(fig)
    plot_3D_dual(P_init, P_opt)
