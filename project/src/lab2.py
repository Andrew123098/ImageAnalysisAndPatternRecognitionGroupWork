from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk, opening
from skimage.transform import rotate, resize
from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import regionprops
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_label(images: np.ndarray, labels: np.ndarray, target_label: int):
    """
    The function returns only the images that have target_label as labels.
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images - handwritten digits 
    labels: np.ndarray (N)
        List of labels associated with the input image
    target_label: int
        Selected target label

    Return
    ------
    img_extract: np.ndarray (M, 28, 28)
        Extracted images that have target_label as label (M should be lower than N).
    """

    n, d, _ = np.shape(images) 
    img_extract = np.zeros((30, d, d))
    
    # ------------------
    indices=np.where(labels==target_label)[0]
    img_extract = images[indices]
    # ------------------
    
    return img_extract


def preprocess(images: np.ndarray):
    """
    Apply the processing step to images to achieve better data uniformity.
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images

    Return
    ------
    img_process: np.ndarray (N, 28, 28)
        Processed images.
    """

    # Get the shape of input data and set dummy values
    n, d, _ = np.shape(images) 
    img_process = np.zeros_like(images)
    
    # ------------------
    img_process = images>=50

    for i in range(n):
        img_process[i]= opening(img_process[i], disk(1.5))
        img_process[i]=closing(img_process[i], disk(2))
    img_process = img_process.astype(np.uint8)
    img_process = img_process * 255
    # ------------------

    return img_process



def find_contour(images: np.ndarray):
    """
    Find the contours for the set of images
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images to process

    Return
    ------
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    """

    # Get number of images to process
    N, _, _ = np.shape(images)
    # Fill in dummy values (fake points)
    contours = [np.array([[0, 0], [1, 1]]) for i in range(N)]
    contour= [np.array([[0, 0], [1, 1]]) for i in range(N)]

    # ------------------
    for i in range(N):
        contours_img=cv2.findContours(images[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_length = 0
        longest_contour = None

        for j in range(len(contours_img[0])):
            length=len(contours_img[0][j])
            if length > max_length:
                longest_contour = contours_img[0][j]
                max_length = length
        contours[i] = longest_contour

        if len(contours_img[0]) > 0:
            for j in range(len(contours_img[0])):
                length = len(contours_img[0][j])
                if length > max_length:
                    longest_contour = contours_img[0][j]
                    max_length = length
            
            # Only update contours[i] and squeeze if a contour was found
            if longest_contour is not None:
                contours[i] = longest_contour
                contours[i] = contours[i].squeeze()  # Now shape is (K, 2)

        if longest_contour is not None:
            contours[i] = longest_contour
            contours[i] = contours[i].squeeze()  # Now shape is (K, 2)
        else:

            print(f"Warning: No valid contour found for image {i}")
            print("Displaying the image:")
            plt.imshow(images[i], cmap='gray')
            plt.title(f"No contour found for image {i}")
            plt.axis('off')
            plt.show()
      # ------------------
    return contours


def compute_descriptor_padding(contours: np.ndarray, n_samples: int = 11):
    """
    Compute Fourier descriptors of input images
    
    Args
    ----
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    n_samples: int
        Number of samples to consider. If the contour length is higher, discard the remaining part. If it is shorter, add padding.
        Make sure that the first element of the descriptor represents the continuous component.

    Return
    ------
    descriptors: np.ndarray complex (N, n_samples)
        Computed complex Fourier descriptors for the given input images
    """

    N = len(contours)
    # Look for the number of contours
    descriptors = np.zeros((N, n_samples), dtype=np.complex128)

    # ------------------
    for i in range(N):
        cont=np.array(contours[i])
        if cont.size == 0:
            continue 
        complex_numbers=np.zeros(len(cont), dtype=np.complex128)
        for j in range(len(cont)):
            complex_numbers[j]=cont[j][0]+1j*cont[j][1]
        if len(cont) > n_samples:
            complex_numbers = complex_numbers[:n_samples]  # Keep only first n_samples points
        else:
            complex_numbers = np.pad(complex_numbers,(0,n_samples-len(complex_numbers)), mode='constant')  # Zero-padding
        descriptors[i]= np.fft.fft(complex_numbers)
    # ------------------

    return descriptors


def linear_interpolation(contours: np.ndarray, n_samples: int = 11):
    """
    Perform interpolation/resampling of the contour across n_samples.
    
    Args
    ----
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    n_samples: int
        Number of samples to consider along the contour.

    Return
    ------
    contours_inter: np.ndarray (N, n_samples, 2)
        Interpolated contour with n_samples
    """

    N = len(contours)
    contours_inter = np.zeros((N, n_samples, 2))
    
    # ------------------
    for i in range(N):
        cont = contours[i]

        t = np.zeros(len(cont))
        for j in range(1, len(cont)):
            t[j] = t[j-1] + np.linalg.norm(cont[j] - cont[j-1]) #linalg->Euclidean norm
        t_max=t[-1]+ np.linalg.norm(cont[0] - cont[-1]) 
    
        t_new = np.array([i * t_max / (n_samples) for i in range(0, n_samples)])
        x_new = np.zeros(n_samples)
        y_new = np.zeros(n_samples)
        for i_new in range(n_samples):
            x_new[i_new] = np.interp(t_new[i_new], t, cont[:, 0])
            y_new[i_new] = np.interp(t_new[i_new], t, cont[:, 1])
        
        contours_inter[i] = np.column_stack((x_new, y_new))
    # ------------------
        
    return contours_inter

def compute_reverse_descriptor(descriptor: np.ndarray, n_samples: int = 11):
    """
    Reverse a Fourier descriptor to xy coordinates given a number of samples.
   
    Args
    ----
    descriptor: np.ndarray (D,)
        Complex descriptor of length D.
    n_samples: int
        Number of samples to consider to reverse transformation.

    Return
    ------
    x: np.ndarray complex (n_samples,)
        x coordinates of the contour
    y: np.ndarray complex (n_samples,)
        y coordinates of the contour
    """

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
   
    # ------------------
    contour = np.fft.ifft(descriptor, n=n_samples)
    
    # The real part corresponds to x coordinates, and the imaginary part to y coordinates
    x = contour.real
    y = contour.imag
    # ------------------

    return x, y


def apply_rotation(img: np.ndarray):
    """
    Apply random rotation to input the image
    
    Args
    ----
    image: np.ndarray (28, 28)
        Source images
        
    Return
    ------
    rotated: np.ndarray (28, 28)
        Rotated source images
    """

    rotated = np.zeros_like(img)
    
    # ------------------
    # Generate a random rotation angle between 0 and 360 degrees
    angle = np.random.uniform(0, 360) 
    rotated = rotate(img, angle, mode='constant', order=1)
    # ------------------
    
    return rotated


def apply_scaling(img: np.ndarray):
    """
    Apply random scaling to input image
    
    Args
    ----
    image: np.ndarray (28, 28)
        Source images
        
    Return
    ------
    scaled: np.ndarray (28, 28)
        Scaled source images
    """
    
    scaled = np.zeros_like(img)
    target_height, target_width = scaled.shape
    
    # ------------------
    scaling_factor = np.random.uniform(1, 1.5)

    resized_img = resize(img, (target_height * scaling_factor, target_width * scaling_factor), anti_aliasing=True)

    start_x = (resized_img.shape[1] - target_width) // 2
    start_y = (resized_img.shape[0] - target_height) // 2

    scaled = resized_img[start_y:start_y + target_height, start_x:start_x + target_width]
    # ------------------
    
    return scaled

def apply_translate(img: np.ndarray):
    """
    Apply random x and y translation to input image
    
    Args
    ----
    image: np.ndarray (28, 28)
        Source images
        
    Return
    ------
    translated: np.ndarray (28, 28)
        Translated source images
    """
    
    translated = np.zeros_like(img)
    
    # ------------------
    random_x = np.random.uniform(-2,2) 
    random_y = np.random.uniform(-2, 2)

    # Get the image dimensions
    rows, cols = img.shape[:2]

    # Create a translation matrix
    M = np.float32([[1, 0, random_x], [0, 1, random_y]])

    # Apply the translation
    translated = cv2.warpAffine(img, M, (cols, rows))
    # ------------------
    
    return translated
def translation_invariant(features):
    """
    Make input Fourier descriptors invariant to translation.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to translation of N images 
        over K (K <= N) features.
    """

    # Set default values
    features_inv = np.zeros_like(features)
    
    # ------------------
    features_inv[:, 1:] = features[:, 1:]
    # ------------------
    
    return features_inv

def rotation_invariant(features):
    """
    Make input Fourier descriptors invariant to rotation.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to rotation of N images 
        over K (K <= N) features.
    """
    
    features_inv = features.copy()
    # ------------------
    # Apply phase correction to all descriptors
    features_inv = np.abs(features_inv)
     # ------------------

    return features_inv


def scaling_invariant(features):
    """
    Make input Fourier descriptors invariant to scaling.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to scaling of N images 
        over K (K <= N) features.
    """
   
    # Set default values
    features_inv = np.zeros_like(features) if features is not None else None
    # ------------------
    # Handle case where features is None
    if features is None:
        print("Warning: features input is None. Returning None.")
        return None
    
    features_inv = features.copy()

    scale_factor = np.abs(features[:, 1])  # Take absolute value to avoid complex numbers

    scale_factor[scale_factor == 0] = 1

    for i in range(features.shape[0]):
        if features[i] is not None:
            features_inv[i, :] = features[i, :] / scale_factor[i]
    # ------------------

    return features_inv


def reference_pattern(imgs):
    """
    Compute the reference pattern for a given set of images. The reference pattern 
    is estimated as the average of all images of the same pattern.

    Args
    ----
    imgs: np.ndarray (N, 28, 28)
        Source images
        
    Return
    ------
    pattern: np.ndarray (28, 28)
        Thresholded reference pattern that is the average of all shapes.
    """

    # Initialize pattern
    pattern = np.zeros((imgs[0].shape[0], imgs[0].shape[1]))
    
    # ------------------
    # The reference pattern is the average image of all images of the same label
    pattern = np.mean(imgs, axis=0)
    # ------------------
   
    return pattern


def compute_distance_map(pattern: np.ndarray):
    """
    Compute the distance map for the given pattern. The values of the map are computed as 
    the distance to the closest pattern contour.

    Args
    ----
    pattern: np.ndarray (28, 28)
        Pattern to process

    Return
    ------
    distance_map: np.ndarray (28, 28)
        Distance map where each entry is the distance to the closest pattern contour (shortest 
        distance to pattern)
    """
    #------------------
    # # Initialize distance map
    distance_map = np.full_like(pattern, np.inf, dtype=np.float32)
    binary_pattern = (pattern > 50).astype(np.uint8)  #keep pattern only

    contours = find_contour(np.expand_dims(binary_pattern, axis=0).astype(np.uint8))
    contour_mask = np.ones_like(binary_pattern, dtype=np.uint8) # Blank image =1

    cv2.drawContours(contour_mask, contours, -1, color=0, thickness=1) # put contours in black-> closest =0

    distance_map = cv2.distanceTransform(contour_mask, distanceType=cv2.DIST_L2, maskSize=5)
    distance_map = cv2.normalize(distance_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
    #-------------------
    
    return distance_map


def compute_distance(imgs, d_map):
    """
    Compute the distances for each image with respect to the reference pattern using the precomputed 
    distance map. The final distance is the average of all distances from the image's contour points 
    to the reference pattern.

    Args
    ----
    imgs: np.ndarray (N, 28, 28)
        Source images
    d_map: np.ndarray (28, 28)
        The precomputed distance map where each entry is the distance to the closest pattern contour 
        (shortest distance to pattern)
    
    Return
    ------
    dist: np.ndarray (N, )
        Averaged distance to pattern for each input image.
    """
    
    # Default values
    dist = np.zeros(len(imgs))

    # ------------------
    # Calculate the distance of every image to the given distance map.
   # Iterate over the images
    for i, img in enumerate(imgs):
        # Find contours for the current image
        contours = find_contour(np.expand_dims(img, axis=0).astype(np.uint8))
        
        # Extract the first contour
        first_contour = contours[0]
        
        # Extract the row and column indices from the contour points
        row_indices = first_contour[:, 0]
        col_indices = first_contour[:, 1]
        
        # Get the distances from the distance map
        distances = d_map[row_indices, col_indices]
        
        # Compute the mean distance
        mean_distance = np.mean(distances)
        
        # Store the mean distance in the dist array
        dist[i] = mean_distance
    # ------------------
    
    return dist


def compute_features(imgs: np.ndarray):
    """
    Compute compacity for each input image.
    
    Args
    ----
    imgs: np.ndarray (N, 28, 28)
        Source images
        
    Return
    ------
    f_peri: np.ndarray (N,)
        Estimated perimeter length for each image
    f_area: np.ndarray (N,)
        Estimated area for each image
    f_comp: np.ndarray (N,)
        Estimated compacity for each image
    f_rect: np.ndarray (N,)
        Estimated rectangularity for each image
    """

    f_peri = np.zeros(len(imgs))
    f_area = np.zeros(len(imgs))
    f_comp = np.zeros(len(imgs))
    f_rect = np.zeros(len(imgs))
    
    # ------------------
    # Calculate the perimeter, area, compacity, and rectangularity of each image.
    for i, img in enumerate(imgs):
        properties = regionprops(img.astype(int))

        f_peri[i] = properties[0].perimeter
        f_area[i] = properties[0].area
        f_comp[i] = f_peri[i]**2 / f_area[i]
        bbox = properties[0].bbox
        f_rect[i] = f_area[i] / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) # Area of the object divided by the area of the bounding box
    # ------------------

    return f_peri, f_area, f_comp, f_rect


