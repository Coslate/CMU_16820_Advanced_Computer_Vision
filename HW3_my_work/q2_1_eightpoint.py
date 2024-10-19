import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here

def computeF(x1, x2):
    # Construct A matrix
    A = np.zeros((x1.shape[0], 9))
    for i in range(x1.shape[0]):
        xi1 = x1[i][0]
        yi1 = x1[i][1]
        xi2 = x2[i][0]
        yi2 = x2[i][1]
        A[i] = [xi1*xi2, xi2*yi1, xi2, yi2*xi1, yi1*yi2, yi2, xi1, yi1, 1]

    # Solve by directly call SVD
    try:
        u, sigma, v = np.linalg.svd(A)
    except np.linalg.LinAlgError as e:
        eps = 1e-10  # Small regularization constant
        perturbation = eps * np.random.rand(A.shape[0], A.shape[1])
        A_regularized = A + perturbation
        u, sigma, v = np.linalg.svd(A_regularized)

    # Choose the eigenvector w/ smallest eigenvalues
    v1_smallest_eigenvec = v[-1, :]
    F2to1_v1_smallest = np.reshape(v1_smallest_eigenvec, (3, 3))
    F2to1_v1 = np.array(F2to1_v1_smallest, dtype=np.float32)

    return F2to1_v1

def normalizeImagePts(pts1, pts2, M):
    T_matrix = np.zeros((3, 3), dtype=np.float32)
    T_matrix[0][0] = 1/M
    T_matrix[0][2] = 0
    T_matrix[1][1] = 1/M
    T_matrix[1][2] = 0
    T_matrix[2][2] = 1

    '''
    T_matrix = np.zeros((3, 3), dtype=np.float32)
    T_matrix[0][0] = 2/M
    T_matrix[0][2] = -1
    T_matrix[1][1] = 2/M
    T_matrix[1][2] = -1
    T_matrix[2][2] = 1
    '''

    x1_homo = np.hstack([pts1, np.ones((pts1.shape[0], 1))]) #Nx3
    x1_norm = T_matrix@x1_homo.T #3xN
    x1_norm = x1_norm.T #Nx3
    x1_col1 = x1_norm[:, 0]/x1_norm[:, 2]
    x1_col2 = x1_norm[:, 1]/x1_norm[:, 2]
    x1_norm = np.column_stack((x1_col1, x1_col2)) #Nx2

    x2_homo = np.hstack([pts2, np.ones((pts2.shape[0], 1))]) #Nx3
    x2_norm = T_matrix@x2_homo.T #3xN
    x2_norm = x2_norm.T #Nx3
    x2_col1 = x2_norm[:, 0]/x2_norm[:, 2]
    x2_col2 = x2_norm[:, 1]/x2_norm[:, 2]
    x2_norm = np.column_stack((x2_col1, x2_col2)) #Nx2

    return T_matrix, x1_norm, x2_norm

"""
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix
"""

def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE

    # Normalizing to [-1, 1] x [-1, 1]
    T_matrix, pts1_norm, pts2_norm = normalizeImagePts(pts1, pts2, M)

    # Setup equations and solve least square solution using SVD
    F_matrix = computeF(pts1_norm, pts2_norm)

    # Enforce singularity condition.
    F_sing = _singularize(F_matrix)

    # Refined the F
    F_refine = refineF(F_sing, pts1_norm, pts2_norm)
    #F_refine = F_sing

    # Unscale the F
    F_unscaled = T_matrix.T @ F_refine @ T_matrix

    # Make sure F is unique avoiding infinite scaling possibilities
    F_ret = F_unscaled/F_unscaled[2, 2]

    return F_ret


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    M = np.max([*im1.shape, *im2.shape])
    F = eightpoint(pts1, pts2, M=M)

    # Q2.1
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1

    # Write file to q2_1.npz 
    np.savez('q2_1.npz', F, M)

    # Load Test
    #loaded_data = np.load('q2_1.npz')
    #print(f"loaded_data['F'] = {loaded_data['F']}")
    #print(f"loaded_data['M'] = {loaded_data['M']}")
