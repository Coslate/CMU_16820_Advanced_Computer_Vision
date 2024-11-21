# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# 
# Nov, 2023
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import (
    loadData,
    estimateAlbedosNormals,
    displayAlbedosNormals,
    estimateShape,
)
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface
import matplotlib.cm as cm
import os

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def plotSurfaceLocal(surface, path="", suffix="", showimg=True):
    """
    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    suffix: str
        suffix for save file

    Returns
    -------
        None

    """
    x, y = np.meshgrid(np.arange(surface.shape[1]), np.arange(surface.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x, y, -surface, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    ax.view_init(elev=60.0, azim=75.0)
    ensure_folder_exists(path)
    plt.savefig(f"{path}/faceCalibrated{suffix}.png")
    if showimg:
        plt.show()

def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    U, Sigma, Vt = np.linalg.svd(I, full_matrices=False)
    Sigma_hat = np.diag(Sigma[:3]) # shape(3, 3)
    U_hat = U[:, :3] #shape(7, 3)
    V_hat =Vt[:3, :] #shape(3, P)

    L_trans = U_hat @ np.sqrt(Sigma_hat) #shape(7, 3)
    B = np.sqrt(Sigma_hat) @ V_hat #shape(3, P)

    # Your code here
    return B, L_trans.T


def plotBasRelief(B, mu, nu, lam, s, path="", suffix="", showimg=True):
    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter

    lambda : float
        bas-relief parameter

    s : tuple
        image shape parameter

    path : str
        path to save images
    
    suffix : std
        saved image name
    
    showing : bool
        whether to show image when running the program

    Returns
    -------
        None

    """
    # Enforce Integrability
    B_ei = enforceIntegrability(B, s)

    # Apply GBR transform on B_ei
    G = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [mu, nu, lam]
    ])

    print(f"mu = {mu}, nu = {nu}, lam = {lam}")
    B_gbr = np.linalg.inv(G).T @ B_ei

    # Plot & Save
    albedos_gbr, normals_gbr = estimateAlbedosNormals(B_gbr)
    surface_gbr = estimateShape(normals_gbr, s)
    plotSurfaceLocal(surface_gbr, path=path, suffix=suffix, showimg=showimg)



def formatted_print(name, array, precision=4):
    """
    Prints a numpy array in a formatted way for easier comparison.
    
    Parameters:
    - name (str): The name of the array to display in the print statement.
    - array (numpy.ndarray): The array to be printed.
    - precision (int): Number of decimal places for formatting. Default is 4.
    """
    formatted_array = np.array2string(array, formatter={'float_kind': lambda x: f"{x:.{precision}f}"})
    print(f"{name} = \n{formatted_array}\n")    

    # Your code here

if __name__ == "__main__":
    I, L, s = loadData("../data/")
    formatted_print('L', L)

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

    # Part 2 (b)
    # Your code here
    B_hat, L_hat = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B_hat)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("2b-a.png", albedoIm, cmap="gray")
    plt.imsave("2b-b.png", normalIm, cmap="rainbow")
    formatted_print('L_hat', L_hat)

    # Part 2 (d)
    # Your code here
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Part 2 (e)
    # Your code here
    est_normals = enforceIntegrability(B_hat, s)
    albedos_ei, normals_ei = estimateAlbedosNormals(est_normals)
    surface_ei = estimateShape(normals_ei, s)
    plotSurface(surface_ei)

    # Part 2 (f)
    # Your code here
    mu_range = [-10, -5, -1, -0.1, -0.001, 0.001, 0.1, 1, 5, 10]
    v_range = [-10, -5, -1, -0.1, -0.001, 0.001, 0.1, 1, 5, 10]
    lambda_range = [-10, -5, -1, -0.1, -0.001, 0.001, 0.1, 1, 5, 10]
    mu_default = 1
    v_default = 1
    lambda_default = 1

    for mu in mu_range:
        plotBasRelief(B_hat, mu, v_default, lambda_default, s, path='./2f_mu_change', suffix=f'_mu_{mu}_v_{v_default}_lambda_{lambda_default}', showimg=False)
    for v in v_range:
        plotBasRelief(B_hat, mu_default, v, lambda_default, s, path='./2f_v_change', suffix=f'_mu_{mu_default}_v_{v}_lambda_{lambda_default}', showimg=False)
    for lambda_ in lambda_range:
        plotBasRelief(B_hat, mu_default, v_default, lambda_, s, path='./2f_lambda_change', suffix=f'_mu_{mu_default}_v_{v_default}_lambda_{lambda_}', showimg=False)

    # Part 2 (g)
    mu_range = [-10, -5, -1, -0.1, -0.001, 0.001, 0.1, 1, 5, 10]
    v_range = [-10, -5, -1, -0.1, -0.001, 0.001, 0.1, 1, 5, 10]
    lambda_range = [-10, -5, -1, -0.1, -0.001, 0.001, 0.1, 1, 5, 10]

    for i in range(len(mu_range)):
        plotBasRelief(B_hat, mu_range[i], v_range[i], lambda_range[i], s, path='./2g_all_change', suffix=f'_mu_{mu_range[i]}_v_{v_range[i]}_lambda_{lambda_range[i]}', showimg=False)

    mu = 10
    v = 1
    lambda_ = 0.001
    plotBasRelief(B_hat, mu, v, lambda_, s, path='./2g_flattest_result', suffix=f'_mu_{mu}_v_{v}_lambda_{lambda_}', showimg=False)
    