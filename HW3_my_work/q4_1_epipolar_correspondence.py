import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here
import cv2

def calculateError(x1_roi: np.ndarray, x2_roi: np.ndarray, kernel: np.ndarray) -> np.float32:
    assert x1_roi.shape == x2_roi.shape

    err = 0
    for channel in range(x1_roi.shape[2]):
        #x1_flat = x1_roi[:, :, channel].reshape(-1)
        #x2_flat = x2_roi[:, :, channel].reshape(-1)
        x1_flat = x1_roi[:, :, channel]
        x2_flat = x2_roi[:, :, channel]
        diff = np.power((x1_flat-x2_flat), 2)
        diff_gauss = applyGaussianKernel(diff, kernel)
        err += np.sum(diff_gauss.reshape(-1))

    return err

def getGaussianWeight(ksize: int=3, sigma: np.float32=1.0, print_verbose: bool=False) -> np.ndarray:
    oneD_kernel = cv2.getGaussianKernel(ksize, sigma=sigma)
    twoD_kernel = oneD_kernel*oneD_kernel.T

    if print_verbose:
        print(twoD_kernel)
        print(twoD_kernel.sum())

    return twoD_kernel

def applyGaussianKernel(roi: np.ndarray, gaussian2DKernel: np.ndarray) -> np.ndarray:
    assert roi.shape == gaussian2DKernel.shape
    ret_result = np.zeros(roi.shape, dtype=np.float32)

    # Apply Gaussian Kernel to each channel
    #for channel in range(roi.shape[2]):
        #ret_result[:, :, channel] = roi[:, :, channel] * gaussian2DKernel
    ret_result[:, :] = roi[:, :] * gaussian2DKernel

    return ret_result

def extractROI(image: np.ndarray, x: int, y: int, ksize:int) -> np.ndarray:
    # Zero padding the image
    # If image is grayscale (2D)
    if len(image.shape) == 2:
        padded_img = np.pad(image, ((ksize//2, ksize//2), (ksize//2, ksize//2)), mode='constant', constant_values=0)
    # If image is color (3D)
    elif len(image.shape) == 3:
        padded_img = np.pad(image, ((ksize//2, ksize//2), (ksize//2, ksize//2), (0, 0)), mode='constant', constant_values=0)
    else:
        raise ValueError("Unsupported image shape")    

    # Padded coordinates
    x_pad = x + ksize//2
    y_pad = y + ksize//2
    x0 = x_pad - ksize//2
    y0 = y_pad - ksize//2
    x1 = x_pad + ksize//2 
    y1 = y_pad + ksize//2

    # Extract ROI
    roi = padded_img[y0 : (y1+1), x0 : (x1+1), :]
    #print(f"(x0, y0) = {(x0, y0)}")
    #print(f"(x1, y1) = {(x1, y1)}")

    return roi

def extractPointsAlongLine(line_vector: np.ndarray, im_width: int, im_height: int):
    ret_points = []
    a, b, c = line_vector[0, 0], line_vector[1, 0], line_vector[2, 0]

    if a and b == 0:
        raise ValueError(f'No valid points due to invalid line_vector: {line_vector}')

    if b == 0:
        x = int(-c/a)
        for y in range(im_height):
            if x >=0 and x < im_width:
                ret_points.append((x, y))
        return ret_points

    if b != 0:
        slope = -a/b
        if np.abs(slope) > 1 and a != 0:
            for y in range(im_height):
            # line equation: x = -(by+c)/a
                x = int(-(b*y+c)/a)
                if x >=0 and x < im_width:
                    ret_points.append((x, y))
        else:
            for x in range(im_width):
            # line equation: y = -(ax+c)/b
                y = int(-(a*x+c)/b)
                if y >=0 and y < im_height:
                    ret_points.append((x, y))
        return ret_points

# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title("Select a point in this image")
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title(
        "Verify that the corresponding point \n is on the epipolar line in this image"
    )
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break

        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            print("Zero line vector in displayEpipolar")

        l = l / s

        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, "*", markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, "ro", markersize=8, linewidth=2)
        plt.draw()


"""
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

"""


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE
    # Construct l2 line equation
    x1_pt = np.array([x1, y1, 1]).reshape(-1, 1)
    l2_line = F @ x1_pt

    # Extract points in im2 along the l2 line 
    x2_pts = extractPointsAlongLine(l2_line, im_width=im2.shape[1], im_height=im2.shape[0])

    # Extract x1_roi
    ksize = 71
    #print(f"---------------------------------------------(x1, y1) = {(x1, y1)}")
    x1_roi = extractROI(image=im1, x=x1, y=y1, ksize=ksize)

    # Get the best pt im im2
    best_err  = np.inf
    best_cand = (None, None)
    for x2_pt_candi in x2_pts:
        #print(f"===========")
        # Extract x2_roi
        x2_roi = extractROI(image=im2, x=x2_pt_candi[0], y=x2_pt_candi[1], ksize=ksize)

        # Apply Gaussian weighting
        gaussien2DKernel = getGaussianWeight(ksize=ksize, sigma=5.0, print_verbose=False)
        #x1_gauss = applyGaussianKernel(x1_roi, gaussian2DKernel=gaussien2DKernel)
        #x2_gauss = applyGaussianKernel(x2_roi, gaussian2DKernel=gaussien2DKernel)

        # Calculate the similarity
        err = calculateError(x1_roi=x1_roi, x2_roi=x2_roi, kernel=gaussien2DKernel)
        #print(f"x2_pt_candi, err = {x2_pt_candi}, {err}")

        # Get the best
        if err < best_err:
            best_err = err
            best_cand = x2_pt_candi

    #x2, y2 = best_cand
    #error = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
    #print(f"error = {error}")
    #print(f"best_cand = {best_cand}")
    return best_cand


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    np.savez("q4_1.npz", F, pts1, pts2)
    epipolarMatchGUI(im1, im2, F)

    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    assert np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10
