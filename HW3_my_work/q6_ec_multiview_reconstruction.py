import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate

# Insert your package here
def triangulate3Points(C1, pts1, C2, pts2, C3, pts3):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE
    # Form Matrix A
    P = np.zeros((pts1.shape[0], 3), dtype=np.float32) # the reconstructed Nx3 3D points
    err = 0.0
    for i in range(pts1.shape[0]):
        A = np.zeros((6, 4), dtype=np.float32)
        uxi1 = pts1[i, 0]
        uyi1 = pts1[i, 1]
        uxi2 = pts2[i, 0]
        uyi2 = pts2[i, 1]
        uxi3 = pts3[i, 0]
        uyi3 = pts3[i, 1]

        A[0] = [C1[0][0]-uxi1*C1[2][0], C1[0][1]-uxi1*C1[2][1], C1[0][2]-uxi1*C1[2][2], C1[0][3]-uxi1*C1[2][3]]
        A[1] = [C1[1][0]-uyi1*C1[2][0], C1[1][1]-uyi1*C1[2][1], C1[1][2]-uyi1*C1[2][2], C1[1][3]-uyi1*C1[2][3]]
        A[2] = [C2[0][0]-uxi2*C2[2][0], C2[0][1]-uxi2*C2[2][1], C2[0][2]-uxi2*C2[2][2], C2[0][3]-uxi2*C2[2][3]]
        A[3] = [C2[1][0]-uyi2*C2[2][0], C2[1][1]-uyi2*C2[2][1], C2[1][2]-uyi2*C2[2][2], C2[1][3]-uyi2*C2[2][3]]
        A[4] = [C3[0][0]-uxi3*C3[2][0], C3[0][1]-uxi3*C3[2][1], C3[0][2]-uxi3*C3[2][2], C3[0][3]-uxi3*C3[2][3]]
        A[5] = [C3[1][0]-uyi3*C3[2][0], C3[1][1]-uyi3*C3[2][1], C3[1][2]-uyi3*C3[2][2], C3[1][3]-uyi3*C3[2][3]]

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
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.

Modified by Vineet Tambe, 2023.
"""

def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=100):
    # TODO: Replace pass by your implementation
    P = np.zeros((pts1.shape[0], 3))
    total_err = 0.0
    valid_points_num = 0

    for i in range(pts1.shape[0]):
        conf1 = pts1[i, 2]
        conf2 = pts2[i, 2]
        conf3 = pts3[i, 2]

        if conf1 > Thres and conf2 > Thres and conf3 > Thres:
            '''
            P_recon_c1c2c3, err_c1c2c3 = triangulate3Points(C1, pts1[i:i+1, :2], C2, pts2[i:i+1, :2], C3, pts3[i:i+1, :2])
            if P_recon_c1c2c3[:, 2] > 0:
                #P[i] = P_recon_c1c2c3
                total_err += err_c1c2c3
                valid_points_num += 1
                P.append(P_recon_c1c2c3)
            '''
            P_recon_c1c2, err_c1c2 = triangulate(C1, pts1[i:i+1, :2], C2, pts2[i:i+1, :2])
            P_recon_c1c3, err_c1c3 = triangulate(C1, pts1[i:i+1, :2], C3, pts3[i:i+1, :2])
            P_recon_c2c3, err_c2c3 = triangulate(C2, pts2[i:i+1, :2], C3, pts3[i:i+1, :2])

            # Average Z > 0 points if applicable
            P_recon_arr = [P_recon_c1c2, P_recon_c1c3, P_recon_c2c3]
            err_arr     = [err_c1c2, err_c1c3, err_c2c3]
            err_avg     = 0
            P_avg = np.zeros((1, 3), dtype=np.float32) # the reconstructed Nx3 3D points
            P_cnt = 0

            for index, P_recon in enumerate(P_recon_arr):
                if P_recon[:, 2] > 0:
                    P_avg += P_recon
                    P_cnt += 1
                    err_avg += err_arr[index]

            if P_cnt == 0:
                continue
            else:
                P[i] = P_avg/P_cnt
                total_err += err_avg/P_cnt
                valid_points_num += 1
        elif conf1 > Thres and conf2 > Thres:
            P_recon_c1c2, err_c1c2 = triangulate(C1, pts1[i:i+1, :2], C2, pts2[i:i+1, :2])
            if P_recon_c1c2[:, 2] > 0:
                P[i] = P_recon_c1c2
                total_err += err_c1c2
                valid_points_num += 1
        elif conf1 > Thres and conf3 > Thres:
            P_recon_c1c3, err_c1c3 = triangulate(C1, pts1[i:i+1, :2], C3, pts3[i:i+1, :2])
            if P_recon_c1c3[:, 2] > 0:
                P[i] = P_recon_c1c3
                total_err += err_c1c3
                valid_points_num += 1
        elif conf2 > Thres and conf3 > Thres:
            P_recon_c2c3, err_c2c3 = triangulate(C2, pts2[i:i+1, :2], C3, pts3[i:i+1, :2])
            if P_recon_c2c3[:, 2] > 0:
                P[i] = P_recon_c2c3
                total_err += err_c2c3
                valid_points_num += 1

    # Calculate the average reprojection error
    if valid_points_num > 0:
        err = total_err
        P = np.array(P).reshape(-1, 3)
    else:
        err = float('inf')  # If no valid points, return infinite error
    return P, err


"""
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
"""


def plot_3d_keypoint_video(pts_3d_video):
    # TODO: Replace pass by your implementation
    """
    plot 3d keypoint
    :param car_points: np.array points * 3
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(10):
        pts_3d = pts_3d_video[i]
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d[index0, 0], pts_3d[index1, 0]]
            yline = [pts_3d[index0, 1], pts_3d[index1, 1]]
            zline = [pts_3d[index0, 2], pts_3d[index1, 2]]
            ax.plot(xline, yline, zline, color=colors[j])

    np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()


# Extra Credit
if __name__ == "__main__":
    pts_3d_video = []
    avg_rep_err = 0
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join("data/q6/", "time" + str(loop) + ".npz")
        image1_path = os.path.join("data/q6/", "cam1_time" + str(loop) + ".jpg")
        image2_path = os.path.join("data/q6/", "cam2_time" + str(loop) + ".jpg")
        image3_path = os.path.join("data/q6/", "cam3_time" + str(loop) + ".jpg")

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data["pts1"]
        pts2 = data["pts2"]
        pts3 = data["pts3"]

        K1 = data["K1"]
        K2 = data["K2"]
        K3 = data["K3"]

        M1 = data["M1"]
        M2 = data["M2"]
        M3 = data["M3"]

        # Note - Press 'Escape' key to exit img preview and loop further
        img = visualize_keypoints(im2, pts2)
        img = visualize_keypoints(im1, pts1)
        img = visualize_keypoints(im3, pts3)

        # TODO: YOUR CODE HERE
        C1 = K1 @ M1
        C2 = K2 @ M2
        C3 = K3 @ M3
        P_recon, err_recon = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=300)
        avg_rep_err += err_recon
        #print(f"reprojection error = {err_recon}")
        plot_3d_keypoint(P_recon)
        pts_3d_video.append(P_recon)


    avg_rep_err /= 10.0
    print(f"The averaged reprojection over 10 frames is {avg_rep_err}")
    np.savez('q6_1.npz', np.array(pts_3d_video).reshape(-1, 3))
    plot_3d_keypoint_video(pts_3d_video)
