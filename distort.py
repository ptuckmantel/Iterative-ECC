import numpy as np
import matplotlib.pyplot as plt
from igor import  binarywave
from glob import glob
import cv2

def generate_transform_xy_single(img, img_orig, offset_guess=[0,0], warp_mode = cv2.MOTION_TRANSLATION, termination_eps = 1e-10,
                          number_of_iterations=10000, gaussfilt=1):
    """
    Determines transformation matrices in x and y coordinates for SingleECC mode

    Parameters
    ----------
    img : cv2
        Currently used image (in cv2 format) to find transformation array of
    img_orig : cv2
        Image (in cv2 format) transformation array is based off of
    offset_guess : list of ints
        Estimated shift and offset between images
    warp_mode : see cv2 documentation
        warp_mode used in cv2's findTransformationECC function
    termination_eps : float
        eps used to terminate fit
    number_of_iterations : int
        number of iterations in fit before termination
    
    Returns
    -------
    warp_matrix : ndarray
        Transformation matrix used to convert img_orig into img
    """
    # Here we generate a MOTION_EUCLIDEAN matrix by doing a 
    # findTransformECC (OpenCV 3.0+ only).
    # Returns the transform matrix of the img with respect to img_orig
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    term_flags = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
    criteria = (term_flags, number_of_iterations, termination_eps)
    warp_matrix[0, 2] = offset_guess[0]
    warp_matrix[1, 2] = offset_guess[1]
    (cc, tform21) = cv2.findTransformECC(img_orig, img, warp_matrix, warp_mode,
                                         criteria, None, gaussfilt)
    return tform21

def array_cropped(array, xoffset, yoffset, offset_caps):
    """
    Crops a numpy_array given the offsets of the array, and the minimum and maximum offsets of a set,
    to include only valid data shared by all arrays

    Parameters
    ----------
    array : array_like
        The array to be cropped
    xoffset : int
        The x-offset ot the array
    yoffset : int
        The y-offset of the array
    offset_caps : list
        A list of four entries. In order, these entries are the xoffset maximum, xoffset
        minimum, yoffset maximum, and yoffset minimum for all arrays

    Returns
    -------
    cropped_array : array_like
    """
    if offset_caps != [0, 0, 0, 0]:
        left = int(np.ceil(offset_caps[0]) - np.floor(xoffset))
        right = int(np.floor(offset_caps[1]) - np.floor(xoffset))
        top = int(np.ceil(offset_caps[2]) - np.floor(yoffset))
        bottom = int(np.floor(offset_caps[3]) - np.floor(yoffset))
        if right == 0:
            right = np.shape(array)[1]
        if bottom == 0:
            bottom = np.shape(array)[0]
        cropped_array = array[top:bottom, left:right]
    else:
        cropped_array = array
    return cropped_array


def array_expanded(array, xoffset, yoffset, offset_caps):
    """
    Expands a numpy_array given the offsets of the array, and the minimum and maximum offsets of a
    set, to include all points of each array. Empty data is set to be NaN

    Parameters
    ----------
    array : array_like
        The array to be expanded
    xoffset : int
        The x-offset ot the array
    yoffset : int
        The y-offset of the array
    offset_caps : list
        A list of four entries. In order, these entries are the xoffset maximum, xoffset
        minimum, yoffset maximum, and yoffset minimum for all arrays

    Returns
    -------
    expanded_array : array_like
        The expanded array
    """
    height = int(np.shape(array)[0] + np.ceil(offset_caps[2]) - np.floor(offset_caps[3]))
    length = int(np.shape(array)[1] + np.ceil(offset_caps[0]) - np.floor(offset_caps[1]))
    expanded_array = np.empty([height, length])
    expanded_array[:] = np.nan
    left = int(-np.floor(offset_caps[1]) + xoffset)
    right = int(length - np.ceil(offset_caps[0]) + xoffset)
    top = int(-np.floor(offset_caps[3]) + yoffset)
    bottom = int(height - np.ceil(offset_caps[2]) + yoffset)
    expanded_array[top:bottom, left:right] = array
    return expanded_array

def full(array, dx, dy):
    h0, w0 = np.shape(array)
    dy = int(dy)
    dx = int(dx)
    expanded_array = np.empty([h0+np.abs(dy), w0+np.abs(dx)])
    expanded_array[:] = np.nan
    if dx >= 0:
        left = dx
        right = dx + w0
    else: 
        left = 0
        right = w0
    if dy >= 0:
        up = dy
        down = dy + h0
    else: 
        up = 0
        down = h0
    expanded_array[up:down , left:right] = array
    return expanded_array
    
def distcorr(z_ref, z_tocorr, imgs_tocorr, gaussfiltsize=1, warp_mode = cv2.MOTION_TRANSLATION):
    matrix = generate_transform_xy_single(z_ref, z_tocorr, gaussfilt=gaussfiltsize, warp_mode = warp_mode)
    xoffsets = np.array(matrix[0, 2])
    yoffsets = np.array(matrix[1, 2])
    print(xoffsets, yoffsets)
    offset_caps = [xoffsets, np.min(xoffsets), np.max(yoffsets), np.min(yoffsets)]
    final_images = [full(i, xoffsets, yoffsets) for i in imgs_tocorr]
    #final_images = [array_cropped(i, xoffsets, yoffsets) for i in imgs_tocorr]
    return final_images, (xoffsets, yoffsets)
