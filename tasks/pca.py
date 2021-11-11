
from scipy.linalg import eigh as largest_eigh
import os
from skimage.io import imread
import numpy as np
import input_output
from skimage.feature import hog


# Compute PCA
def compute_pca(data, k=-1):
    # Compute features of every image
    print("Computing PCA...")
    data_matrix = np.array(data)
    covariance = np.cov(data_matrix.transpose())
    # print(covariance.shape)
    print("Covariance matrix size is: ",len(covariance))
    evals_large, evecs_large = largest_eigh(covariance, eigvals=(len(covariance) - k, len(covariance) - 1))
    # print(evecs_large.shape)
    print(f"{evecs_large.shape} - evecs large shape")
    left_factor_matrix = np.matmul(data_matrix, evecs_large)
    right_factor_matrix = np.transpose(evecs_large)
    print(f"{left_factor_matrix.shape} - left fatcor matrix shape")
    print(f"{right_factor_matrix.shape} - right fatcor matrix shape")
    return left_factor_matrix, right_factor_matrix
