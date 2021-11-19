import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib

from PIL import Image


# get the k similar images, returned value is list of file_name, image
def get_k_similar_images(inputImage, images, file_names, k):
    colorMomentInputImage = compute_color_moment_of_image(inputImage)
    pairs = []
    k_similar = []
    values = []
    for i in range(len(images)):
        colorMomentReferredImage = compute_color_moment_of_image(images[i])
        pairs.append((find_color_moment_similarity(colorMomentInputImage, colorMomentReferredImage),
                      (file_names[i], images[i])))
    pairs.sort(key=lambda i: i[0], reverse=False)
    for i in range(len(pairs)):
        k_similar.append((pairs[i][1][0], pairs[i][1][1]))
        values.append((pairs[i][1][0], pairs[i][0]))
    return k_similar[:k], values


# calculate value denoting the similarity between two images using Manhattan distance
def find_color_moment_similarity(cmImg1, cmImg2):
    mean1 = cmImg1[0]
    mean2 = cmImg2[0]
    sd1 = cmImg1[1]
    sd2 = cmImg2[1]
    sk1 = cmImg1[2]
    sk2 = cmImg2[2]
    diff1 = 0
    diff2 = 0
    diff3 = 0
    for i in range(0, len(mean1)):
        diff1 += abs(mean1[i] - mean2[i])
        diff2 += abs(sd1[i] - sd2[i])
        diff3 += abs(sk1[i] - sk2[i])

    return diff1 + diff2 + diff3


# this function will return the color moment of an image, in the format 64*3,
# 0th index array is mean, 1st index value is deviation and 2nd index value is skewness
def compute_color_moment_of_image(image):
    featureDescription = []
    featureDescription.append([])
    featureDescription.append([])
    featureDescription.append([])
    imageSize = (64, 64)
    counterx = 8
    countery = 8
    count = 0

    while counterx <= 64 and countery <= 64:
        count += 1
        nparray = get_window(counterx - 8, countery - 8, counterx, countery, image)
        mean = np.mean(nparray)
        deviation = np.std(nparray)
        skewness = scipy.stats.skew(nparray, axis=None)
        featureDescription[0].append(mean)
        featureDescription[1].append(deviation)
        featureDescription[2].append(skewness)

        if (countery >= 64):
            counterx += 8
            countery = 0
        countery += 8
    return featureDescription


def calculate_skewness(nparray, mean):
    skew = 0
    for i in range(len(nparray)):
        for j in range(len(nparray[i])):
            skew += np.power((nparray[i][j] - mean), 3)
    skew /= 64
    return np.cbrt(skew)


# creating the 8x8 window for color moments
def get_window(x, y, highx, highy, image):
    arr = []
    for i in range(x, highx):
        val = []
        for j in range(y, highy):
            val.append(image[i][j])
        arr.append(val)
    x = np.array(arr)
    return x
