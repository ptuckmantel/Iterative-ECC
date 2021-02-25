import numpy as np
import cv2

def to_gray(img):
    """ Convert array of floats to array 8bit integers"""
    img = np.array(img)
    img=(img-np.nanmin(img))
    img = img/np.nanmax(img)
    img=img*255
    img = img.astype(np.uint8)

    return img


def get_offset(wm):
    """Extract x,y offsets from warp matrix"""
    x = wm[0, 2]
    y = wm[1,2]
    
    return y,x

def set_offset(wm, dy, dx):
    """Sets x,y offsets for warp matrix"""
    wm[0, 2] = dx
    wm[1,2] = dy
    return wm


def estimate_correction_quality(corrected, reference):
    """
    Calculates the Frobenius norm of (corrected - reference) normalised by the number of non-nan pixels.
    First replaces 0s in corrected with Nans
    
    corrected: distortion-corrected scan (2d array)
    reference: reference scan used for correction (2d array)
    
    returns: pixel-averaged norm (float)
    """
    #convert dtypes to allow replacing cropped areas (with value 0) with nans
    corrected = corrected.astype(float)
    reference = reference.astype(float)
    # calculate the difference between corrected and reference image, with cropped area set to nan
    # so it is not included in the difference map
    corrected[corrected == 0] = np.nan
    diff = corrected - reference

    numpix = np.sum(~np.isnan(diff))
    # Assign 0 to the nan values in diff so they do not contribute to frobenius norm
    diff [ np.isnan(diff)] = 0
    # Calculate the frobenius norm to
    norm = np.linalg.norm(diff)/numpix
    
    return norm


def try_corr(z_ref, z_tocorr, warp_matrix):
    """
    Attempts an ECC correction
    
    z_ref: reference topography (2d array of of dtype=int8)
    z_tocorr: topography to correct (2d array of dtype=int8)
    
    returns: 
    F: Pixel-averaged Frobenius norm of (corrected_scan - reference_scan)
    img_corr: corrected image (2d array dtype=int8)
    warp_matrix: warp matrix describing the correction
    """
    dx = warp_matrix[0,2]
    dy = warp_matrix[1,2]
    sz = z_ref.shape

    warp_matrix = warp_matrix.astype('float32')
    
    if np.shape(warp_matrix) == (3,3):
        warp_mode = cv2.MOTION_HOMOGRAPHY
    elif np.shape(warp_matrix) == (2,3):
        warp_mode = cv2.MOTION_TRANSLATION
    else:
        print('Invalid warp matrix shape. Cropping matrix and correcting with CV2.MOTION_TRANSLATION')
        warp_matrix = warp_matrix[:2, :3]
        warp_mode = cv2.MOTION_TRANSLATION
    
    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    

    try:
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (z_ref,z_tocorr,warp_matrix, warp_mode, criteria, None, 1)

        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography 
            img_corr= cv2.warpPerspective (z_tocorr, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            img_corr = cv2.warpAffine(z_tocorr, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        F = estimate_correction_quality(img_corr, z_ref)
        #print('Correction successful')
        
        return F, img_corr, warp_matrix

    except:
        #print('Correction failed')
        
        return np.nan, np.nan, np.nan

    

def ecc_1step(z_ref, z_tocorr, shifts, initial_shift):
    """
    Perform ecc correction for a set of vertical and horizontal shifts around an initial shift
    
    z_ref: reference topography (2d array of of dtype=int8)
    z_tocorr: topography to correct (2d array of dtype=int8)
    shifts: list of (y, x) shift values to attempt
    initial_shift: (y, x) center value around from which the values in shifts are offset
    
    returns: 
    F: Lowest pixel-averaged Frobenius norm of (corrected_scan - reference_scan) among attempts (float)
    img_corr: corrected image of best attempt (2d array dtype=int8)
    warp_matrix: warp matrix describing the correction of best attempt (2d array)
    """
    Fs, imgs_corr, warp_matrices = [], [], []
    warp_matrix = np.eye(2,3, dtype = np.float32)

    
    for s in shifts:
        dy = s[0] + initial_shift[0]
        dx = s[1] + initial_shift[1]
        #print('Attempting dy: %.5f, dx: %.5f' %(dy, dx))
        # Define input warp matrix with specified y,x shifts
        warp_matrix = set_offset(warp_matrix, dy, dx)
        # Attempt a distortion correction
        F, img_corr, wmc = try_corr(z_ref, z_tocorr, warp_matrix)
        
            
        # Append F, corrected image and correction warp matrix to corresponding arrays of attempts
        Fs.append(F)
        imgs_corr.append(img_corr)
        warp_matrices.append(wmc)
            
    # Get the coordinates of the shift with the lowest F value   
    ind_opt = np.nanargmin(Fs)
    
    return Fs[ind_opt], imgs_corr[ind_opt], warp_matrices[ind_opt]
        
    
    
def get_warp_iterative_ecc(z_ref, z_tocorr, shifts, max_steps = 5, thresh = 2, verbose = False):
    """Applies iterative EEC andd returns optimal correction error, corrected image and warp matrix

    z_ref: reference toography (2d array of of dtype=int8)
    z_tocorr: topography to correct (2d array of dtype=int8)
    shifts: list of [y, x] shift values to attempt
    max_steps: maximum number of steps in iterative ecc (int, default: 5)
    thresh: threshold euclidean distance between the  shifts in two consecutive ECC tries
    
    Returns:
    F: Lowest pixel-averaged Frobenius norm of (corrected_scan - reference_scan) among attempts (float)
    img_corr: corrected image of best attempt (2d array dtype=int8)
    warp_matrix: warp matrix describing the correction of best attempt (2d array)
    
    """
    initial_shift = [0, 0]
    Fs, warp_matrices, imgs_corr = [], [], []

    
    # Attempt correction from initial position and append F, corrected image and correction warp matrix
    warp_matrix = np.eye(2,3, dtype = np.float32)
    #print('Attempting dy: %.5f, dx: %.5f' %(0, 0))
    F, img_corr, wmc = try_corr(z_ref, z_tocorr, warp_matrix)
    Fs.append(F)
    imgs_corr.append(img_corr)
    warp_matrices.append(wmc)
    
    if ~np.isnan(F):
        offset_ini = get_offset(wmc)
        if verbose:
            print('initial offset after correction: (%.5f, %.5f)' %(offset_ini[0], offset_ini[1]))
    else:
        if verbose:
            print('correction attempt from initial image coordinates failed')
    
        for step in range(max_steps):
            if verbose:
                print('=== iteration %d ===' %(step+1))
            try:
                
                offset_ini = [0,0]
                F, img_corr, wmc = ecc_1step(z_ref, z_tocorr, shifts, offset_ini)
                Fs.append(F)
                imgs_corr.append(img_corr)
                warp_matrices.append(wmc)
                new_offset = get_offset(wmc)
                dist = np.linalg.norm(np.asarray(new_offset)-np.asarray(initial_shift))
                if verbose:
                    print('best shift: (%.5f, %.5f)' %(new_offset[0], new_offset[1]))
                    print('Euclidean distance with last attempt: %.5f' %dist)
                if dist > thresh:
                    initial_shift = new_offset
                else:
                    if verbose:
                        print('offset difference lower than threshold. Stopping now')
                    break
            except:
                if verbose:
                    print('no correction attempt succeeded. Stopping now')
                break

    if Fs:
        ind_opt = np.nanargmin(Fs)
        return Fs[ind_opt], imgs_corr[ind_opt], warp_matrices[ind_opt]
    
    else:
        return np.nan, np.nan, np.nan
    
    
def apply_warp(tocorr, warp_matrix):
    """ Apply a warping to input array specified by input warp_matrix
    
    tocorr: array to warp
    warp_matrix: matrix describing the transformation
    
    returns: corrected image of same shape as input
    """
    sz = tocorr.shape

    warp_matrix = warp_matrix.astype('float32')
    
    if np.shape(warp_matrix) == (3,3):
        warp_mode = cv2.MOTION_HOMOGRAPHY
    elif np.shape(warp_matrix) == (2,3):
        warp_mode = cv2.MOTION_TRANSLATION
    else:
        print('Invalid warp matrix shape. Cropping matrix and correcting with CV2.MOTION_TRANSLATION')
        warp_matrix = warp_matrix[:2, :3]
        warp_mode = cv2.MOTION_TRANSLATION
        
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        img_corr= cv2.warpPerspective (tocorr, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        img_corr = cv2.warpAffine(tocorr, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
 
    return img_corr
