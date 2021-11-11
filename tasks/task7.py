from .task4 import get_type_feature_matrix, get_subject_weight_matrix
#from .distance_functions import manhattan, mahalonobis
import numpy as np
from numpy import genfromtxt
from input_output import get_image_arr_from_file
from .task5 import get_query_image_in_latent_space
from scipy.spatial import distance
from .task5 import task5


# Entry point for task 7 to find the subject id
def assign_tag_to_new_query_image(query_image_path, feature_model, latent_semantics_file_path):
    # subject_weight_matrix, keys = get_subject_weight_matrix('Dataset', feature_model)
    # print("keys")
    # print(keys)
    # latent_semantics = np.matrix(genfromtxt(latent_semantics_file_path, delimiter=','))
    # query_image = get_image_arr_from_file(query_image_path)
    # query_image_new = get_query_image_in_latent_space(query_image, latent_semantics, feature_model)
    # subject_matrix_new_space = np.matmul(np.matrix(subject_weight_matrix), np.matrix.transpose(latent_semantics))
    # min_dist = 100000000
    # tag = -1
    # for i in range(len(subject_matrix_new_space)):
    #     dist = manhattan(query_image_new, subject_matrix_new_space[i])
    #     if min_dist > dist:
    #         min_dist = dist
    #         tag = i
    # return keys[tag]
    # TODO pass 1 as k
    image = task5(query_image_path, latent_semantics_file_path, 1, feature_model)
    image_name = image[len(image)-1][1]
    return image_name.split('-')[2];
