import numpy as np


# Get the weight matrix.
def get_weight_matrix(images_with_attributes, left_factor_matrix, search='subject_id'):
    search_left_factor_matrix = {}
    for n in range(len(images_with_attributes)):
        if(images_with_attributes[n][search] in search_left_factor_matrix):
            search_left_factor_matrix[images_with_attributes[n][search]].append(left_factor_matrix[n])
        else:
            image_arrays = []
            image_arrays.append(left_factor_matrix[n])
            search_left_factor_matrix[images_with_attributes[n][search]] = image_arrays
    
    search_weight_dict = {}
    for key, value in search_left_factor_matrix.items():
        array_lol = np.array(value)
        average = np.average(array_lol, axis=0)
        search_weight_dict[key] = average
    
    search_weight_matrix = []

    for key in search_weight_dict.keys():
        search_weight_matrix.append(search_weight_dict[key])
    return search_weight_matrix, list(search_weight_dict.keys())


def get_subject_weight_matrix(images_with_attributes, left_factor_matrix):
    print('Getting Subject-Weight matrix')
    matrix, _ = get_weight_matrix(images_with_attributes, left_factor_matrix, 'subject_id')
    return matrix


def get_type_weight_matrix(images_with_attributes, left_factor_matrix):
    print('Getting Type-Weight matrix')
    matrix, _= get_weight_matrix(images_with_attributes, left_factor_matrix, 'type')
    return matrix