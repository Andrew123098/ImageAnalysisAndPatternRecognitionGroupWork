import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Callable
from datetime import datetime
from skimage.color import rgb2hsv
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects, binary_dilation



def extract_rgb_channels(img):
    """
    Extract RGB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_red: np.ndarray (M, N)
        Red channel of input image
    data_green: np.ndarray (M, N)
        Green channel of input image
    data_blue: np.ndarray (M, N)
        Blue channel of input image
    """

    # Get the shape of the input image
    M, N, _ = np.shape(img)

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    # ------------------
    # Extract the RGB channels
    data_red = img[:, :, 0]
    data_green = img[:, :, 1]
    data_blue = img[:, :, 2]
    # ------------------
    return data_red, data_green, data_blue

def apply_rgb_threshold(img):
    """
    Apply threshold to input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract RGB channels
    data_red, data_green, data_blue = extract_rgb_channels(img=img)
    
    # ------------------
    RGB = extract_rgb_channels(img)
    R_th = 40 # Greater than
    G_th = 95  # Less than
    B_th = 165 # Less than
    img_th = (RGB[0] > R_th) & (RGB[1] < G_th) & (RGB[2] < B_th)
    # ------------------
    
    return  img_th


def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for HSV channels
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    # ------------------
    hsv_img = rgb2hsv(img)
    data_h= hsv_img[:,:,0]
    data_s= hsv_img[:,:,1]
    data_v = hsv_img[:,:,2]
    # ------------------
    
    return data_h, data_s, data_v

def apply_hsv_threshold(img, H_min=0, H_max=1, S_min=0, S_max=1, V_min=0, V_max=1):
    """
    Apply threshold to the input image in hsv colorspace.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract HSV channels
    data_h, data_s, data_v = extract_hsv_channels(img=img)
    
    # ------------------
    img_H = (data_h > H_min) & (data_h < H_max) #hue filter
    img_S = (data_s > S_min) & (data_s < S_max) #saturation filter
    img_V = (data_v > V_min) & (data_v < V_max) #value filter
    img_th = img_H & img_S & img_V #combine filters
    # ------------------
    
    return  img_th

def apply_closing(img_th, disk_size):
    """
    Apply closing to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_closing: np.ndarray (M, N)
        Image after closing operation
    """

    # Define default value for output image
    img_closing = np.zeros_like(img_th)
    disk_size=disk_size
    # ------------------
    img_closing=closing(img_th, disk(disk_size))
    # ------------------

    return img_closing



def apply_opening(img_th, disk_size):
    """
    Apply opening to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_opening: np.ndarray (M, N)
        Image after opening operation
    """

    # Define default value for output image
    img_opening = np.zeros_like(img_th)

    # ------------------
    img_opening=opening(img_th, disk(disk_size)) 
    # ------------------

    return img_opening

def remove_holes(img_th, size):
    """
    Remove holes from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of holes

    Return
    ------
    img_holes: np.ndarray (M, N)
        Image after remove holes operation
    """

    # Define default value for input image
    img_holes = np.zeros_like(img_th)

    # ------------------
    img_holes = remove_small_holes(img_th, area_threshold=size, connectivity=1, out=None)
    # ------------------

    return img_holes


def remove_objects(img_th, size):
    """
    Remove objects from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of objects

    Return
    ------
    img_obj: np.ndarray (M, N)
        Image after remove small objects operation
    """

    # Define default value for input image
    img_obj = np.zeros_like(img_th)

    # ------------------
    img_obj = remove_small_objects(img_th, min_size=size, connectivity=1, out=None)
    # ------------------

    return img_obj


def apply_morphology(img_th, kernel_size=4, min_area_th=500, min_obj_size=250, connect=1):
    """
    Apply morphology to thresholded image

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.

    Return
    ------
    img_morph: np.ndarray (M, N)
        Image after morphological operations
    """

    img_morph = np.zeros_like(img_th)
    
    # ------------------
    img_morph = opening(img_morph, disk(kernel_size))
    img_morph = closing(img_th, disk(kernel_size))
    img_morph = remove_small_holes(img_morph, area_threshold=min_area_th, connectivity=connect, out=None)
    img_morph = remove_small_objects(img_morph, min_size=min_obj_size, connectivity=connect, out=None)
    img_morph = closing(img_th, disk(kernel_size))
    img_morph = opening(img_morph, disk(kernel_size))
    # ------------------
    return img_morph



def region_growing(
    seeds: list[tuple],
    img: np.ndarray,
    n_max: int = 10,
    **kwargs
):
    """
    Run region growing on input image using seed points.

    Args
    ----
    seeds: list of tuple
        List of seed points
    img: np.ndarray (M, N, C)
        RGB image of size M, N, C
    n_max: int
        Number maximum of iterations before stopping algorithm

    Return
    ------
    rg: np.ndarray (M, N)
        Image after region growing has been performed
    """
    
    M, N, _ = img.shape
    rg = np.zeros((M, N)).astype(bool)
    
    # ------------------
    # Apply the morphological operations to the input image to get the mask
    mask = apply_morphology(apply_hsv_threshold(img))

    # Mark seed points in the region mask
    rows, cols = zip(*seeds)  # Unpack the list of tuples into rows and columns
    rg[rows, cols] = True  # Mark seed points as Foreground (True)

    # Define the structuring element for 8-connected neighbors
    kernel = np.ones((3, 3)).astype(bool)  # 3x3 kernel for 8-connected neighbors

    # Region Growing
    iterations = 0
    while iterations < n_max:
        # Store the current region before updating
        rg_old = rg

        # Perform binary dilation with the mask
        rg = binary_dilation(image=rg,footprint=kernel)

        # Mask the result with the input image to ensure we only grow within the foreground
        rg = rg & mask

        # Stop if no new pixels are added
        if np.array_equal(rg, rg_old):
            break

        iterations += 1
    # ------------------
                    
    return rg