import os
import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, Callable
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import LedoitWolf


class MahalanobisClassifier:
    """Mahalanobis based classifer"""

    def __init__(self):
        """
        Attributes:
            means (torch.tensor): (n_classes, d) Mean of the features for each class
            inv_covs (torch.tensor): (n_classes, d, d) Inverse of covariance matrix across d features for each class   
        """
        super().__init__()
        self.means = None
        self.inv_covs = None
        
    def fit(self, train_x : torch.Tensor, train_y : torch.Tensor):
        """Computes parameters for Mahalanobis Classifier (self.mean and self.cov), fitted on the training data.

        Args:
            train_x (torch.Tensor): (N, d) The tensor of training features
            train_y (torch.Tensor): (N,) The tensor of training labels
        """

        # Define number of classes
        n_classes = len(np.unique(np.unique(train_y)))
        n, d = train_x.shape
        
        # Set default values
        means = torch.zeros((n_classes, d), dtype=train_x.dtype)
        inv_covs = torch.ones((n_classes, d, d), dtype=train_x.dtype)
        
        # ------------------
        # Convert to numpy for sklearn compatibility
        train_x_np = train_x.numpy()
        train_y_np = train_y.numpy()

        classes = np.unique(train_y_np)

        for i, c in enumerate(classes):
            # Get samples for the curent class
            class_samples = train_x_np[train_y_np == c]

            # Compute class mean
            means[i] = torch.from_numpy(np.mean(class_samples, axis=0))

            # Compute class covariance matrix
            lw = LedoitWolf().fit(class_samples)

            cov_matrix = lw.covariance_

            # Compute inverse covariance matrix
            inv_covs[i] = torch.from_numpy(np.linalg.pinv(cov_matrix))
        
        # ------------------
            
        self.means = means
        self.inv_covs = inv_covs



    def predict(self, test_x : torch.Tensor) -> torch.Tensor:
        """Predicts the class of every test feature, using the Mahalanobis Distance

        Args:
            test_x (torch.Tensor): (N, d) The tensor of test features

        Returns:
            preds (torch.Tensor): (N,) The predictions tensor (id of the predicted class {0, 1, ..., n_classes-1})
            dists (torch.Tensor): (N, n_classes) Mahalanobis distance from sample to class means
        """

        # Define default output value
        N, d = test_x.shape
        dists = torch.zeros((N, self.means.shape[0]), dtype=test_x.dtype)
        preds = torch.zeros(N, dtype=test_x.dtype)
        
        # ------------------
        # Compute Mahalanobis distance for each test sample to each of the classes
        for i in range(N): 
            x = test_x[i] # Current test sample
            for j in range(self.means.shape[0]):

                # Calculate Difference Vector
                diff = x - self.means[j]

                # Calculate Mahalanobis distance
                dist = torch.sqrt(diff @ self.inv_covs[j] @ diff.T)
                
                # Store distance
                dists[i, j] = dist

            # Get the class with the minimum distance
            preds[i] = torch.argmin(dists[i])
        # ------------------

        return preds, dists
    
#     mahalanobis_classifier(
#     MahalanobisClassifier, train_x, train_y, val_x, val_y,
#     cls_name=["Tumor", "Stroma"], colors=["r", "b"],
# )

class MahalanobisOODClassifier(MahalanobisClassifier):
    """Predicts the class of every test feature, using the Mahalanobis Distance

    Args:
        test_x (torch.Tensor): (N x d) The tensor of test features

    Returns:
        preds (torch.Tensor): (N,) The predictions tensor (id of the predicted class {0, 1, ..., n_classes-1})
        dists (torch.Tensor): (N, n_classes) Mahalanobis distance from sample to class means
        ood_scores (torch.Tensor): (N,) Score of OoDness as the minimal distance from the sample to classes
    """

    def predict(self, test_x : torch.Tensor) -> torch.Tensor:
        
        # Get super prediction (from MahalanobisClassifier)
        preds, dists = super().predict(test_x=test_x)
        N = preds.shape[0]

        # Assign dummy values to scores
        ood_scores = np.zeros(N)
        
        # ------------------
         # Compute OOD scores (minimum distance across all classes) as the OoDness score
        ood_scores = torch.min(dists, dim=1)[0]
        # ------------------
        
        return preds, dists, ood_scores
# classifier_ood, val_y_ood_scores=mahalanobis_ood_classifier(
#     MahalanobisOODClassifier, train_x, train_y, val_x, val_y,
#     cls_name=["Tumor", "Stroma"], colors=["r", "b"],
# )

def get_ood_threshold(ood_scores, quantile=0.95):
    """ Get OoD threshold based on measured scores and quantile

    Args:
        ood_scores (torch.Tensor): (N, ) N measured OoDness scores
        quantile (float): Percentage of samples that are considered as in distribution
    """

    # Set default value
    threshold = 0

    # ------------------
    # Compute threshold based on quantile
    threshold = np.quantile(ood_scores, quantile)
    # ------------------

    return threshold


def compute_metrics(y, y_hat, ood_scores, threshold):
    """ Compute recall for tumor, stroma, and OoD as well as the average recall.

    Args:
        y (torch.Tensor): (N) Class ground truth {-1, 0, 1, ..., n_classes}
        y_hat (torch.Tensor): (N,) Class predictions {0, 1, ..., n_classes}
        ood_scores (torch.Tensor): (N, ) N measured OoDness scores
        threshold (float): OoD threshold
    """
    # Define variable with dummy values 
    recall_tumor = 0
    recall_stroma = 0
    recall_ood = 0
    avg_recall = 0
    
    # ------------------
    # Convert all inputs to numpy if they're tensors
    y_np = y.numpy() if torch.is_tensor(y) else y
    y_hat_np = y_hat.numpy() if torch.is_tensor(y_hat) else y_hat
    ood_scores_np = ood_scores.numpy() if torch.is_tensor(ood_scores) else ood_scores
    threshold_np = threshold.item() if torch.is_tensor(threshold) else threshold
    
    # Mark predictions as OOD (-1) if their score exceeds threshold
    y_hat_ood = np.where(ood_scores_np > threshold_np, -1, y_hat_np)
    
    # Calculate recalls
    recall_tumor = np.mean((y_hat_ood == 0) & (y_np == 0)) / np.mean(y_np == 0)
    recall_stroma = np.mean((y_hat_ood == 1) & (y_np == 1)) / np.mean(y_np == 1)
    recall_ood = np.mean((y_hat_ood == -1) & (y_np == -1)) / np.mean(y_np == -1)
    
    # Handle potential division by zero
    recall_tumor = 0 if np.isnan(recall_tumor) else recall_tumor
    recall_stroma = 0 if np.isnan(recall_stroma) else recall_stroma
    recall_ood = 0 if np.isnan(recall_ood) else recall_ood
    
    # Calculate average recall
    avg_recall = (recall_tumor + recall_stroma + recall_ood) / 3

    # ------------------

    return recall_tumor, recall_stroma, recall_ood, avg_recall

class kNNClassifier:
    """k-NN based classifier"""

    def __init__(self, k : int):
        """
        Args:
            k (int): The number of neighbors to consider for the classification
            features (torch.Tensor): (N, d) feature of the N train samples
            labels (torch.Tensor): (N,) labels for train samples
        """
        self.k = k
        self.features = None
        self.labels = None

    def fit(self, train_x : torch.Tensor, train_y : torch.Tensor):
        """Store training data parameters (features and labels) for k-NN classifier.

        Args:
            train_x (torch.Tensor): (N, d) The tensor of training features
            train_y (torch.Tensor): (N,) The tensor of training labels
        """
        
        # Get size and default values
        N, d = train_x.shape
        features = torch.zeros((N, d))
        labels = torch.zeros(N)
        
        # ------------------
        # Simply store the training data
        features = train_x.clone()
        labels = train_y.clone()
        # ------------------

        self.features = features
        self.labels = labels

    def predict(self, test_x: torch.Tensor) -> torch.Tensor:
        """Predicts the class of every test feature, using the k-NN"""
        N, d = test_x.shape
        preds = torch.zeros(N, dtype=torch.long)
        ood_scores = torch.zeros(N)
        
        # ------------------
        # Compute distances between test and train features
        distances = torch.cdist(test_x, self.features)
        
        # Get k nearest neighbors for each test sample
        _, indices = torch.topk(distances, k=self.k, largest=False, dim=1)

        # Get labels of nearest neighbors
        neighbor_labels = self.labels[indices]

        # Get the most common label among the neighbors (majority vote)
        if self.k == 1:
            preds = neighbor_labels.squeeze(1)
        else:
            # Get number of classes (convert max to int first)
            n_classes = int(torch.max(self.labels).item()) + 1
            counts = torch.zeros((N, n_classes), dtype=torch.long)
            
            for i in range(N):
                unique, cnt = torch.unique(neighbor_labels[i], return_counts=True)
                counts[i, unique.long()] = cnt
            preds = torch.argmax(counts, dim=1)
    
        # OOD score: average distance to k nearest neighbors
        ood_scores = torch.mean(distances.gather(1, indices), dim=1)
        # ------------------
        
        return preds, ood_scores
    

def find_best_k(ks,kNNClassifier: Callable,train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor):
    best_k = 0
    best_accuracy = 0.
    # Iterate over ks
    for k in ks:

        # ------------------
        classifier = kNNClassifier(k=k)
        classifier.fit(train_x, train_y)
        preds, _ = classifier.predict(val_x)
        
        accuracy = accuracy_score(val_y.numpy(), preds.numpy())
        print(f"Accuracy for k={k}: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k  # Fixed: assign current k, not the whole list
        # ------------------
        
        continue

    return best_k, best_accuracy

def fit_knn(best_k, train_x, train_y, val_x, val_y):

    # best threshold
    threshold_val = 0
    # Predicted val ood scores
    val_y_ood_scores = torch.zeros(len(val_y))
    classifier = None 
    # ------------------
    # efine the classifier object
    classifier = kNNClassifier(k=best_k)

    # First fit the classifier on training data
    classifier.fit(train_x, train_y)
    
    # Get OOD scores for validation set
    _, val_y_ood_scores = classifier.predict(val_x)
    
    # Compute threshold for 95% ID retention
    threshold_val = torch.quantile(val_y_ood_scores, 0.95)
    # ------------------

    return classifier, threshold_val, val_y_ood_scores