import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
import skimage.feature
from scipy.ndimage import rotate

PATCHWIDTH = 9

def briefMatch(desc1,desc2,ratio):

    matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True,max_ratio=ratio)
    return matches



def plotMatches(im1,im2,matches,locs1,locs2):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.axis('off')
    skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r',only_matches=True)
    plt.show()
    return



def makeTestPattern(patchWidth, nbits):

    np.random.seed(0)
    compareX = patchWidth*patchWidth * np.random.random((nbits,1))
    compareX = np.floor(compareX).astype(int)
    np.random.seed(1)
    compareY = patchWidth*patchWidth * np.random.random((nbits,1))
    compareY = np.floor(compareY).astype(int)

    return (compareX, compareY)




def computePixel(img, idx1, idx2, width, center, index, length):

    halfWidth = width // 2
    col1 = idx1 % width - halfWidth
    row1 = idx1 // width - halfWidth
    col2 = idx2 % width - halfWidth
    row2 = idx2 // width - halfWidth
    return 1 if img[int(center[0]+row1)][int(center[1]+col1)] < img[int(center[0]+row2)][int(center[1]+col2)] else 0


def computePixelRotInv(img, idx1, idx2, width, center, orientation, index, length):
    #rotated_image = rotate(img, angle=-1*orientation, reshape=True)
    #orientation_rad = -orientation*np.pi/180

    halfWidth = width // 2
    col1 = idx1 % width - halfWidth
    row1 = idx1 // width - halfWidth
    col2 = idx2 % width - halfWidth
    row2 = idx2 // width - halfWidth
    y = center[0]
    x = center[1]
    patch = img[max(0, y-halfWidth):min(img.shape[0], y+halfWidth+1),
                max(0, x-halfWidth):min(img.shape[1], x+halfWidth+1)]    

    rotated_patch = rotate(patch, angle=-1*orientation, reshape=True)
#   return 1 if img[row1_idx][col1_idx)] < img[row2_idx][col2_idx] else 0
    return 1 if rotated_patch[int(row1)][int(col1)] < rotated_patch[int(row2)][int(col2)] else 0


def computeBrief(img, locs):

    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern(patchWidth,nbits)
    m, n = img.shape

    halfWidth = patchWidth//2

    locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
    desc = np.array([list(map(lambda x: computePixel(img, x[0], x[1], patchWidth, c, index, len(locs)), zip(compareX, compareY))) for index, c in enumerate(locs)])

    return desc, locs

def computeBriefRotInv(img, locs, orientation):

    patchWidth = 9
    nbits = 256

    compareX, compareY = makeTestPattern(patchWidth,nbits)
    m, n = img.shape

    halfWidth = patchWidth//2

    locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
    desc = np.array([list(map(lambda x: computePixelRotInv(img, x[0], x[1], patchWidth, c, orientation[index], index, len(locs)), zip(compareX, compareY))) for index, c in enumerate(locs)])

    return desc, locs


def corner_detectionRotInv(img, sigma):

    # fast method
    result_img = skimage.feature.corner_fast(img, n=PATCHWIDTH, threshold=sigma)
    locs = skimage.feature.corner_peaks(result_img, min_distance=1)
    corner_values = result_img[locs[:, 0],  locs[:, 1]]
    sorted_indices = np.argsort(corner_values)[::-1]  # Sort in descending order
    best_500_corners = locs[sorted_indices[:500]]  # Get the top 500 corners

    return best_500_corners


def corner_detection(img, sigma):

    # fast method
    result_img = skimage.feature.corner_fast(img, n=PATCHWIDTH, threshold=sigma)
    locs = skimage.feature.corner_peaks(result_img, min_distance=1)
    return locs


def loadVid(path):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    cap = cv2.VideoCapture(path)

    # Append frames to list
    frames = []

    # Check if camera opened successfully
    if cap.isOpened()== False:
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            #Store the resulting frame
            frames.append(frame)
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    frames = np.stack(frames)

    return frames
