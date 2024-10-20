import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here

def Falpha(alpha: np.float32, F_v1_smallest: np.ndarray, F_v2_smallest: np.ndarray) -> np.ndarray:
    return alpha*F_v1_smallest + (1-alpha)*F_v2_smallest

def FalphaDet(alpha: np.float32, F_v1_smallest: np.ndarray, F_v2_smallest: np.ndarray) -> np.ndarray:
    return np.linalg.det(alpha*F_v1_smallest + (1-alpha)*F_v2_smallest)

def calCoeff(F_v1_smallest: np.ndarray, F_v2_smallest: np.ndarray) -> np.ndarray :
    # Create det of F(alpha)
    F_p0      = FalphaDet(0 , F_v1_smallest, F_v2_smallest)
    F_p1      = FalphaDet(1 , F_v1_smallest, F_v2_smallest)
    F_n1      = FalphaDet(-1, F_v1_smallest, F_v2_smallest)
    F_p2      = FalphaDet(2 , F_v1_smallest, F_v2_smallest)
    F_n2      = FalphaDet(-2 , F_v1_smallest, F_v2_smallest)


    # Calculate the coefficients for the polynomial equation
    '''
    coefficients = np.zeros(4)
    coefficients[0] = F_p0
    coefficients[2] = (F_p1+F_n1)/2.0 - coefficients[0]
    coefficients[3] = (F_p3 - 9*coefficients[2] - coefficients[0])/15 - (F_p2 - 4*coefficients[2] - coefficients[0])/10
    coefficients[1] = F_p1 - (coefficients[0]+coefficients[2]+coefficients[3])
    '''
    coefficients = np.zeros(4)
    coefficients[0] = F_p0
    coefficients[2] = (F_p1+F_n1)/2.0 - coefficients[0]
    coefficients[3] = (F_p2-F_n2)/12.0 - (F_p1-F_n1)/6.0
    coefficients[1] = (F_p1-F_n1)/2.0 - coefficients[3]

    return coefficients

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

    v1_smallest_eigenvec = v[-1, :]
    v2_smallest_eigenvec = v[-2, :]

    F2to1_v1_smallest = np.reshape(v1_smallest_eigenvec, (3, 3))
    F2to1_v2_smallest = np.reshape(v2_smallest_eigenvec, (3, 3))
    F2to1_v1 = np.array(F2to1_v1_smallest, dtype=np.float32)
    F2to1_v2 = np.array(F2to1_v2_smallest, dtype=np.float32)

    return F2to1_v1, F2to1_v2

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
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
"""


def sevenpoint(pts1, pts2, M):
    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE

    # Normalizing to [0, 1] x [0, 1]
    T_matrix, pts1_norm, pts2_norm = normalizeImagePts(pts1, pts2, M)

    # Setup matrix and solve SVD and pick the last two columns
    F_v1_matrix, F_v2_matrix = computeF(pts1_norm, pts2_norm)

    # Calculate c4-c0
    coefficients = calCoeff(F_v1_matrix, F_v2_matrix)

    # Solve for the roots
    roots = np.polynomial.polynomial.polyroots(list(coefficients))

    # Setup equations and solve least square solution using SVD
    F_alpha_array = [Falpha(alpha, F_v1_matrix, F_v2_matrix) for alpha in roots]

    # Enforce singularity condition.
    F_sing_array = []
    for F_matrix in F_alpha_array:
        try:
            F_sing = _singularize(F_matrix)
        except np.linalg.LinAlgError as e:
            F_sing = F_matrix.copy()
        F_sing_array.append(F_sing)
    #F_sing_array = [_singularize(F_matrix) for F_matrix in F_alpha_array]
    #F_sing_array = [F_matrix for F_matrix in F_alpha_array]

    # Refined the F
    F_refine_array = []
    for F_sing in F_sing_array:
        try:
            F_refine = refineF(F_sing, pts1_norm, pts2_norm)
        except np.linalg.LinAlgError as e:
            F_refine = F_sing.copy()
        F_refine_array.append(F_refine)
    #F_refine_array = [refineF(F_sing, pts1_norm, pts2_norm) for F_sing in F_sing_array]
    #F_refine_array = [F_sing for F_sing in F_sing_array]

    # Unscale the F
    F_unscaled_array = [T_matrix.T @ F_refine @ T_matrix for F_refine in F_refine_array]

    # Make sure F is unique avoiding infinite scaling possibilities
    Farray = [F_unscaled/F_unscaled[2, 2] for F_unscaled in F_unscaled_array]

    return Farray


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    F = Farray[0]

    #np.savez("q2_2.npz", F, M)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    #displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            if np.isnan(res).any(): continue
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1

    displayEpipolarF(im1, im2, F)
    np.savez("q2_2.npz", F, M)
