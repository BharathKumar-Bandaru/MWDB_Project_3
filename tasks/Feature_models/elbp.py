import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from skimage.feature import local_binary_pattern
from PIL import Image


# using the skimage library to calculate the local binary pattern, method passed is "uniform" which integrates LBP
# with rotational variance.
def compute_elbp(img):
    x = local_binary_pattern(img, 16, 2, "uniform") + local_binary_pattern(img, 8, 1,"uniform") + local_binary_pattern(img, 24, 3, "uniform")
    return x.flatten()

# using Manhattan Distance to calculate the distance between two Local Binary Patterns
def lpb_comparison_manhanttan(lbp1, lbp2):
    diff1 = 0
    for i in range(0, len(lbp1)):
        for j in range(0, len(lbp1[i])):
            diff1 += abs(lbp1[i][j] - lbp2[i][j])
    return diff1


# function to calculate the difference between the local binary pattern's of two images
def lbp_distance_of_images(image1, image2):
    lbp1 = compute_elbp(image1)
    lbp2 = compute_elbp(image2)
    return lpb_comparison_manhanttan(lbp1, lbp2)


# get k similar images according to the ELBP model, returned value is list of file_name, image
def get_k_similar_images(img, images, file_names, k):
    lbp1 = compute_elbp(img)
    pairs = []
    k_similar = []
    values = []
    for i in range(len(images)):
        lbp2 = compute_elbp(images[i])
        x = lpb_comparison_manhanttan(lbp1, lbp2)
        pairs.append((x, (file_names[i], images[i])))

    pairs.sort(key=lambda i: i[0], reverse=False)
    for i in range(len(pairs)):
        k_similar.append((pairs[i][1][0], pairs[i][1][1]))
        values.append((pairs[i][1][0], pairs[i][0]))

    return k_similar[:k], values
