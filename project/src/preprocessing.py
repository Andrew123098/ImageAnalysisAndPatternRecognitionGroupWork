import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import cv2
from typing import Literal


def load_reference_images(path):
    """Load all of the reference images from the given path and return the images as a list of numpy arrays
        with the shape (MxNx3) where M and N are the dimensions of the image
        and the corresponding labels as a list of strings from the filenames which are the labels ex. 'chocolate.jpg' -> 'chocolate'
    Args:
        path (str): Path to the reference images directory.
    Returns:
        images (list): Array of loaded images.
        labels (list): List of labels corresponding to the images.
        images_dict (dict): Dictionary mapping labels to images.
    """
    images = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.JPG'):
            # print(f"Loading image: {filename}")
            img = Image.open(os.path.join(path, filename))
            images.append(np.array(img))
            labels.append(filename.split('.')[0])

    images_dict = {label: img for label, img in zip(labels, images)}
    return images, labels, images_dict

def plot_reference_images(images, labels):
    """Plot the reference images with their labels in one big figure.
    
    Args:
        images (list): List of images to plot.
        labels (list): List of labels corresponding to the images.
    """
    if len(images) != len(labels):
        raise ValueError("Number of images and labels must match")
        
    num_images = len(images)
    rows = int(np.ceil(num_images / 3))  # Flexible row calculation
    fig, axs = plt.subplots(rows, 3, figsize=(15, 4.5*rows))
    axs = axs.ravel()  # Flatten the array for easy indexing
    
    for i in range(num_images):
        axs[i].imshow(images[i])
        axs[i].set_title(labels[i])
        axs[i].axis('off')
        
    # Hide any unused subplots
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_reference_image(images_dict, label):
    """Plot the reference image with its label."""
    plt.imshow(images_dict[label])
    plt.title(label)
    plt.axis('off')
    plt.show()

def load_train_images(path):
    """Load all of the train images from the given path and return the images as a list of numpy arrays
        with the shape (MxNx3) where M and N are the dimensions of the image
        and the corresponding labels as a list of strings from the filenames which are the labels ex. 'chocolate.jpg' -> 'chocolate'
    Args:
        path (str): Path to the reference images directory.
    Returns:
        images (list): Array of loaded images.
        labels (list): List of labels corresponding to the images.
        images_dict (dict): Dictionary mapping labels to images.
    """
    images = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.JPG'):
            # print(f"Loading image: {filename}")
            img = Image.open(os.path.join(path, filename))
            images.append(np.array(img))
            labels.append(filename.split('.')[0])

    # Create a dictionary with labels as keys and nested dictionaries as values
    images_dict = {
        label: {
            "image": img,
            "R": None,
            "G": None,
            "B": None
        }
        for label, img in zip(labels, images)
    }
    return images, labels, images_dict


def downsample_images(images, dictionay, downsample_factor=0.1):
    """Downsample the images by the given factor and update the dictionary with the downsampled images.
    
    Args:
        images (list): List of images to downsample.
        dictionay (dict): Dictionary to update with downsampled images.
        downsample_factor (int): Factor by which to downsample the images.
        
    Returns:
        None: The function updates the dictionary in place.
    """
    for label, img in zip(dictionay.keys(), images):
        downsampled_img = img[::downsample_factor, ::downsample_factor]
        dictionay[label]['image'] = downsampled_img




def downsample_image(
    img: np.ndarray,
    method: Literal["grid", "kmeans", "stratified"] = "grid",
    **kwargs
) -> np.ndarray:
    """
    Unified downsampling interface with three strategies.
    
    Args:
        img: Input RGB image (H,W,3)
        method: Downsampling strategy ('grid', 'kmeans', or 'stratified')
        **kwargs: Strategy-specific parameters:
            - grid: grid_size
            - kmeans: n_colors
            - stratified: n_samples
    
    Returns:
        Downsampled RGB image array
    """
    strategy_map = {
        "grid": _grid_downsample,
        "kmeans": _kmeans_downsample,
        "stratified": _stratified_downsample
    }
    
    return strategy_map[method](img, **kwargs)

def _grid_downsample(
    img: np.ndarray,
    target_height: int = None,
    target_width: int = None,
    grid_size: int = None,
    preserve_aspect: bool = True
) -> np.ndarray:
    """
    Aspect-ratio-preserving grid downsampling.
    
    Args:
        img: Input image (H,W,3)
        target_height: Desired output height (optional)
        target_width: Desired output width (optional)
        grid_size: Alternative to specify square grid size
        preserve_aspect: Maintain original aspect ratio
    
    Returns:
        Downsampled image with preserved aspect ratio
    """
    h, w = img.shape[:2]
    
    # Calculate grid dimensions
    if grid_size:
        grid_h = grid_size
        grid_w = grid_size
    elif target_height and target_width:
        if preserve_aspect:
            # Calculate closest dimensions that preserve aspect
            aspect = w / h
            if (target_width / target_height) > aspect:
                target_width = int(target_height * aspect)
            else:
                target_height = int(target_width / aspect)
        grid_h = h // target_height
        grid_w = w // target_width
    else:
        raise ValueError("Must specify either grid_size or target dimensions")
    
    # Ensure grids divide evenly
    grid_h = max(1, grid_h)
    grid_w = max(1, grid_w)
    new_h = h // grid_h
    new_w = w // grid_w
    
    # Trim and downsample
    trimmed = img[:new_h*grid_h, :new_w*grid_w]
    downsampled = trimmed.reshape(
        new_h, grid_h, new_w, grid_w, 3
    ).mean(axis=(1, 3)).astype(np.uint8)
    
    # print(f"Downsampled from {img.shape} to {downsampled.shape}")
    return downsampled

def _kmeans_downsample(
    img: np.ndarray,
    n_colors: int = 256
) -> np.ndarray:
    """
    Color quantization using K-means clustering.
    
    Args:
        img: Input image (H,W,3)
        n_colors: Number of target colors
    
    Returns:
        Quantized image (H,W,3) with reduced colors
    """
    pixels = img.reshape(-1, 3)
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=0,
        batch_size=1024
    )
    labels = kmeans.fit_predict(pixels)
    return kmeans.cluster_centers_[labels].reshape(img.shape).astype(np.uint8)

def _stratified_downsample(
    img: np.ndarray,
    n_samples: int = 1000,
    n_preclusters: int = 100
) -> np.ndarray:
    """
    Stratified color sampling preserving distribution.
    
    Args:
        img: Input image (H,W,3)
        n_samples: Total samples to return
        n_preclusters: Initial clustering for stratification
    
    Returns:
        Array of sampled pixels (n_samples, 3) in RGB
    """
    # Convert to Lab for better color distance
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3)
    
    # Initial clustering for stratification
    kmeans = MiniBatchKMeans(
        n_clusters=min(n_preclusters, len(pixels)),
        random_state=0
    )
    labels = kmeans.fit_predict(pixels)
    
    # Sample proportionally from each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    samples_per_cluster = np.ceil(
        n_samples * counts / counts.sum()
    ).astype(int)
    
    sampled_pixels = []
    for label, n in zip(unique_labels, samples_per_cluster):
        cluster_pixels = pixels[labels == label]
        if len(cluster_pixels) > n:
            selected = cluster_pixels[
                np.random.choice(len(cluster_pixels), n, replace=False)
            ]
        else:
            selected = cluster_pixels
        sampled_pixels.append(selected)
    
    # Convert back to RGB
    sampled_lab = np.vstack(sampled_pixels)
    sampled_rgb = cv2.cvtColor(
        sampled_lab.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_LAB2RGB
    )
    return sampled_rgb.reshape(-1, 3)

def compare_2_images(image1, image2, title, labels=None):
    """Compare two images side by side with optional labels.
    
    Args:
        image1 (np.ndarray): First image to compare.
        image2 (np.ndarray): Second image to compare.
        title (str): Title for the comparison plot.
        labels (list, optional): List of labels for the images. Defaults to None.
    """
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    # plt.title(labels[0] if labels else "Image 1")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    # plt.title(labels[1] if labels else "Image 2")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

def RGB2greyscale(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using luminosity method.
    
    Args:
        img: Input RGB image (H,W,3)
    
    Returns:
        Grayscale image (H,W)
    """
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]