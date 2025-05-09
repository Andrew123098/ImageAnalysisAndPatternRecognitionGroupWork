import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Callable
from datetime import datetime


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
    rows = int(np.ceil(num_images / 7))  # Flexible row calculation
    fig, axs = plt.subplots(rows, 7, figsize=(15, 2.5*rows))
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
    if label in images_dict:
        plt.imshow(images_dict[label])
        plt.title(label)
        plt.axis('off')
        plt.show()
    else:
        print(f"Label '{label}' not found in the images dictionary.")


