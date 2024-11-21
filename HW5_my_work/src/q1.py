
# Carnegie Mellon University
# Nov, 2023
###################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2xyz
from utils import plotSurface, integrateFrankot

from skimage import io
import scipy.io


def renderNDotLSphere(center, rad, light, pxSize, res):
    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0] / 2) * pxSize * 1.0e-4
    Y = (Y - res[1] / 2) * pxSize * 1.0e-4
    Z = np.sqrt(rad**2 + 0j - X**2 - Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)
    image = None
    # Your code here
    # Identify the visible region on the hemisphere
    visible = Z > 0  # Only points within the bowl's radius in positive Z

    # Calculate surface normals for the visible points on the hemisphere
    Nx = X[visible] - center[0]
    Ny = Y[visible] - center[1]
    Nz = Z[visible] - center[2]
    normals = np.stack((Nx, Ny, Nz), axis=-1)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    
    # Calculate the dot product (n dot l) for Lambertian lighting
    ndotl = np.dot(normals, light)
    ndotl[ndotl < 0] = 0  # Ignore negative values (light coming from behind)
    
    # Create the final image with lighting applied
    image = np.zeros((res[1], res[0]))
    image[visible] = ndotl/np.max(ndotl)*255

    
    return image    



def loadData(path="../data/"):
    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None
    # Your code here
    I = []

    # Load images and convert to luminance channel
    for n in range(1, 8):
        image = io.imread(f"{path}/input_{n}.tif").astype(np.uint16)
        assert image.dtype == np.uint16

        xyz_image = rgb2xyz(image)
        luminance = xyz_image[:, :, 1]  # Extract Y channel (luminance)
        #luminance = (luminance*65535).astype(np.uint16)
        if s is None:
            s = luminance.shape
        I.append(luminance.flatten())
    I = np.array(I)

    # Load the sources file
    L = np.load(f"{path}/sources.npy").T

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):
    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    # Your code here
    B = np.linalg.inv(L@L.T)@(L@I)
    return B


def estimateAlbedosNormals(B):
    """
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    """
    eps = 1e-6

    albedos = np.linalg.norm(B, axis=0) # (P,)
    normals = B/(albedos+eps) #eps: avoid dividing by zero
    # Your code here
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    """
    Question 1 (f, g)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape(s)
    normalIm = normals.T.reshape((s[0], s[1], 3)) #(3, P) -> (P, 3) -> (h, w, 3)
    normalIm = (normalIm + 1.0)/2.0 # from value range [-1, 1] to [0, 1]
    # Your code here
    return albedoIm, normalIm


def estimateShape(normals, s):
    """
    Question 1 (j)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    # Your code here
    # Reshape normals into the image shape
    n1 = normals[0, :].reshape(s)  # X-component of normals, (h, w, 1)
    n2 = normals[1, :].reshape(s)  # Y-component of normals, (h, w, 1)
    n3 = normals[2, :].reshape(s)  # Z-component of normals, (h, w, 1)

    # Compute gradients f_x and f_y
    fx = -n1 / n3
    fy = -n2 / n3

    # Use utils.integrateFrankot for integration
    surface = integrateFrankot(fx, fy)    
    return surface


if __name__ == "__main__":
    # Part 1(b)
    radius = 0.75  # cm
    center = np.asarray([0, 0, 0])  # cm
    pxSize = 7  # um
    res = (3840, 2160)

    light = np.asarray([1, 1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("1b-a.png", image, cmap="gray")

    light = np.asarray([1, -1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("1b-b.png", image, cmap="gray")

    light = np.asarray([-1, -1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("1b-c.png", image, cmap="gray")


    # Part 1(c)
    I, L, s = loadData("../data/")
    #print(f"I.shape = {I.shape}")
    #print(f"s = {s}")
    print(f"L = {L}")
    
    '''
    # test code
    print(f"L = {L}")
    print(f"I[0].dtype = {I[0].dtype}")

    num_images = I.shape[0]
    for i in range(num_images):
        image = I[i].reshape(s)  # Reshape to original shape
        plt.imshow(image, cmap='gray')
        plt.title(f"Image {i+1}")
        plt.axis('off')
        plt.show()
    '''

    # Part 1(d)
    # Your code here
    u_vec, singular, vt_vec = np.linalg.svd(I, full_matrices=False)
    print("Singular Values of I:", singular)

    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)
    #print(f"B.shape = {B.shape}")

    # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("1f-a.png", albedoIm, cmap="gray")
    plt.imsave("1f-b.png", normalIm, cmap="rainbow")

    # Part 1(i)
    surface = estimateShape(normals, s)
    plotSurface(surface)
