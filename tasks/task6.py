from .task3 import get_type_feature_matrix, get_type_weight_matrix
from .task5 import get_query_image_in_latent_space
from .distance_functions import manhattan
import numpy as np
from numpy import genfromtxt
from input_output import get_image_arr_from_file


# Entry point for task 6 to find the correct type.
def assign_tag_to_new_query_image(query_image_path, feature_model, latent_semantics_file_path):
    type_weight_matrix, keys = get_type_weight_matrix('Dataset', feature_model)
    latent_semantics = np.matrix(genfromtxt(latent_semantics_file_path, delimiter=','))
    query_image = get_image_arr_from_file(query_image_path)
    query_image_new = get_query_image_in_latent_space(query_image, latent_semantics, feature_model)
    type_matrix_new_space = np.matmul(np.matrix(type_weight_matrix), np.matrix.transpose(latent_semantics))
    min_dist = 100000000
    tag = -1
    for i in range(len(type_matrix_new_space)):
        dist = manhattan(query_image_new, type_matrix_new_space[i])
        if min_dist > dist:
            min_dist = dist
            tag = i
    return keys[tag]

