import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


"""
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
"""


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE
    # Form Matrix A
    P = np.zeros((pts1.shape[0], 3), dtype=np.float32) # the reconstructed Nx3 3D points
    err = 0.0
    for i in range(pts1.shape[0]):
        A = np.zeros((4, 4), dtype=np.float32)
        uxi1 = pts1[i, 0]
        uyi1 = pts1[i, 1]
        uxi2 = pts2[i, 0]
        uyi2 = pts2[i, 1]

        A[0] = [C1[0][0]-uxi1*C1[2][0], C1[0][1]-uxi1*C1[2][1], C1[0][2]-uxi1*C1[2][2], C1[0][3]-uxi1*C1[2][3]]
        A[1] = [C1[1][0]-uyi1*C1[2][0], C1[1][1]-uyi1*C1[2][1], C1[1][2]-uyi1*C1[2][2], C1[1][3]-uyi1*C1[2][3]]
        A[2] = [C2[0][0]-uxi2*C2[2][0], C2[0][1]-uxi2*C2[2][1], C2[0][2]-uxi2*C2[2][2], C2[0][3]-uxi2*C2[2][3]]
        A[3] = [C2[1][0]-uyi2*C2[2][0], C2[1][1]-uyi2*C2[2][1], C2[1][2]-uyi2*C2[2][2], C2[1][3]-uyi2*C2[2][3]]

        # Solve by directly call SVD
        try:
            u, sigma, v = np.linalg.svd(A)
        except np.linalg.LinAlgError as e:
            eps = 1e-10  # Small regularization constant
            perturbation = eps * np.random.rand(A.shape[0], A.shape[1])
            A_regularized = A + perturbation
            u, sigma, v = np.linalg.svd(A_regularized)

        v1_smallest_eigenvec = v[-1, :].reshape(1, -1) # the solution of wi
        P[i] = v1_smallest_eigenvec[:, 0:3] / v1_smallest_eigenvec[0, -1] #back from homogeneous coordinates to normal coordinates


    # Calculat the error
    w_homo = np.hstack([P, np.ones((P.shape[0], 1))]) #Nx4
    w1_proj = C1@w_homo.T #3x4*4xN
    w1_proj = w1_proj.T #Nx3
    x1_col1 = w1_proj[:, 0]/w1_proj[:, 2]
    x1_col2 = w1_proj[:, 1]/w1_proj[:, 2]
    x1  = np.column_stack((x1_col1, x1_col2)) #Nx2
    w2_proj = C2@w_homo.T #3x4*4xN
    w2_proj = w2_proj.T #Nx3
    x2_col1 = w2_proj[:, 0]/w2_proj[:, 2]
    x2_col2 = w2_proj[:, 1]/w2_proj[:, 2]
    x2  = np.column_stack((x2_col1, x2_col2)) #Nx2
    err = np.sum(np.linalg.norm(pts1-x1, axis=1)**2) + np.sum(np.linalg.norm(pts2-x2, axis=1)**2)

    return P, err

"""
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
"""


def findM2(F, pts1, pts2, intrinsics, filename="q3_3.npz"):
    """
    Q2.2: Function to find camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track
        of the projection error through best_error and retain the best one.
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'.

    """
    # ----- TODO -----
    # YOUR CODE HERE
    # Calculate E from F and K1/K2
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    E = essentialMatrix(F, K1, K2)

    # Calcualte M1 & M2
    M1 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)  #[I|0]
    M2s = camera2(E)
    C1 = K1 @ M1

    # Select the M2 that has most positive depth points
    max_posdepth_num = -1
    #min_error = np.inf
    M2_ret = M2s[:, :, 0]
    P_ret  = np.zeros((pts1.shape[0], 3))
    for index in range(M2s.shape[-1]):
        M2 = M2s[:, :, index]
        C2 = K2 @ M2
        P, error = triangulate(C1, pts1, C2, pts2)

        '''
        if error < min_error:
            min_error = error
            M2_ret = M2.copy()
            P_ret = P.copy()
        '''
        posdepth_num = np.sum((P[:, 2] > 0))
        if posdepth_num > max_posdepth_num:
            max_posdepth_num = posdepth_num
            M2_ret = M2.copy()
            P_ret = P.copy()

    C2_ret = K2 @ M2_ret

    # Write file to q3_3.npz 
    np.savez(filename, M2_ret, C2_ret, P_ret)
    return M2_ret, C2_ret, P_ret

if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    #print(f"err = {err}")
    assert err < 500
