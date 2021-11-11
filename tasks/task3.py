from input_output import get_images_and_attributes_from_folder, store_array_as_csv
from .weight_matrices_calc import get_weight_matrix
from .simitary_matrices import get_simitary_matrices
from .features import compute_features
from .dim_red import perform_dim_red
from .task1_2 import sort_print_matrix
import numpy as np

# task 3, we need to pass folder path, k latent semantics, feature model
def get_type_feature_matrix(folder_path, k, feature_model, dim_reduction):
    type_weight_matrix, keys = get_type_weight_matrix(folder_path, feature_model, dim_reduction)
    print("type keys:")
    print(keys)
    type_type_similarity = get_simitary_matrices(type_weight_matrix)
    store_array_as_csv(type_type_similarity, 'output', 'task3_type_type_similarity.csv')
    left_matrix, right_matrix = perform_dim_red(dim_reduction, type_type_similarity, k)
    store_array_as_csv(left_matrix, 'output', 'task3_left_matrix.csv')
    store_array_as_csv(right_matrix, 'output', 'task3_latent_semantics.csv')
    sort_print_matrix(left_matrix, k, "t")
    return left_matrix


# Get the type weight matrix
def get_type_weight_matrix(folder_path, feature_model, dim_reduction="elbp"):
    images_attributes = get_images_and_attributes_from_folder(folder_path)
    images = []
    for attr in images_attributes:
        images.append(attr["image"])
    images_features = []
    for img in images:
        images_features.append(compute_features(img, feature_model))
    if feature_model == "cm" and dim_reduction == "lda":
        feature_max = np.max(images_features)
        images_features = images_features + feature_max

    return get_weight_matrix(images_attributes, images_features, "type")







