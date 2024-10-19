import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

# Insert your package here

def on_key(event):
    """Handle key press events."""
    if event.key == 'escape':  # ESC key
        print("ESC pressed! Exiting...")
        plt.close()  # Close the plot

"""
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2200 on the 3D points. 

    Modified by Vineet Tambe, 2023.
"""


def compute3D_pts(temple_pts1, intrinsics, F, im1, im2, filename='q4_2.npz'):
    # ----- TODO -----
    # YOUR CODE HERE
    # Calcualte M1 & C1
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    M1 = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)  #[I|0]
    C1 = K1 @ M1
    # Get corresponding point [x2, y2]
    temple_pts2 = np.zeros(temple_pts1.shape, dtype=int)
    for index, (x1, y1) in enumerate(temple_pts1):
        temple_pts2[index] = epipolarCorrespondence(im1, im2, F, x1, y1)

    # Computer M2 matrix
    M2, C2, P = findM2(F, temple_pts1, temple_pts2, intrinsics, filename="q4_2_3_3.npz")
    
    # Write file to filename
    np.savez(filename, F, M1, M2, C1, C2)
    return P


def plot_3D(P):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2])

    # Connect the key press event to the function on_key
    fig.canvas.mpl_connect('key_press_event', on_key)

    while plt.fignum_exists(fig.number):
        clicked_points = plt.ginput(1, mouse_stop=2)

        if len(clicked_points) > 0:
            x, y = clicked_points[0]
            print(f"Clicked at: {x}, {y}")
        plt.draw()
    print("Exited the loop and program closed.")

"""
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
"""
if __name__ == "__main__":
    temple_coords = np.load("data/templeCoords.npz")
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # Call compute3D_pts to get the 3D points and visualize using matplotlib scatter
    temple_pts1 = np.hstack([temple_coords["x1"], temple_coords["y1"]])

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    P = compute3D_pts(temple_pts1, intrinsics, F, im1, im2)

    # Visualize
    fig = plt.figure()
    ax = Axes3D(fig)
    plot_3D(P)
