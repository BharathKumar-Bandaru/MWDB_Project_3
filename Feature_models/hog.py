import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image
from skimage import feature
from skimage.transform import rescale, resize


# using the hog function of skimage library to calculate the gradient
def computer_hog(image):
    hog_image_rescaled= feature.hog(resize(image, (128, 64)), orientations=9, pixels_per_cell=(8, 8),
                                     cells_per_block=(2, 2), block_norm='L2', visualize=False, transform_sqrt=False,
                                     feature_vector=True, multichannel=None)
    return hog_image_rescaled


# using Manhattan distance to calculate the distance between two images using the HOG model
def hog_manhattan(img1, img2):
    img1_resized = img1.ravel()
    img2_resized = img2.ravel()
    sum = 0
    for i in range(len(img1_resized)):
        sum += abs(img1_resized[i] - img2_resized[i])
    return sum


# get k similar images according to the ELBP model, returned value is list of file_name, image
def get_k_similar_images(img, images, file_names, k):
    hog_input_image = computer_hog(img)
    k_similar = []
    values = []
    pairs = []
    for i in range(len(images)):
        hog_referred_image = computer_hog(images[i])
        x = hog_manhattan(hog_input_image, hog_referred_image)
        pairs.append((x, (file_names[i], images[i])))

    pairs.sort(key=lambda i: i[0], reverse=False)
    for i in range(len(pairs)):
        k_similar.append((pairs[i][1][0], pairs[i][1][1]))
        values.append((pairs[i][1][0], pairs[i][0]))
    return k_similar[:k], values
