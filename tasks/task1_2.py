from input_output import get_images_with_attributes, get_image_arr_from_dict, store_array_as_csv
from .features import *
from .pca import compute_pca
from .svd import compute_svd
from .kmeans import kmeans
from .lda import LDA
from .dim_red import perform_dim_red

from .weight_matrices_calc import get_subject_weight_matrix, get_type_weight_matrix


# Entry for the task 1 and 2
def task1_2(feature_model, filter, image_type, k, dim_red_technique,
            folder_path='input_images', output_folder='output',
            latent_semantics_file_name='task1_latent_semantics.csv'):
    images_with_attributes = get_images_with_attributes(folder_path, filter=filter, filter_value=image_type)
    images = get_image_arr_from_dict(images_with_attributes)
    image_features = get_flattened_features_for_images(images, feature_model)
    if feature_model == "cm" and dim_red_technique == "lda":
        feature_max_value = np.max(image_features)
        image_features = image_features + feature_max_value
    left_factor_matrix = core_matrix = right_factor_matrix = None

    dim_red_technique = dim_red_technique.lower()
    left_factor_matrix, right_factor_matrix = perform_dim_red(dim_red_technique, image_features, k)
    perform_post_operations(images_with_attributes, left_factor_matrix, right_factor_matrix, output_folder,
                            latent_semantics_file_name, filter)


# Perform operations to save the latent semantics
def perform_post_operations(images_with_attributes, left_factor_matrix, right_factor_matrix, output_folder,
                            latent_semantics_file_name, filter, 
                            subject_weight_matrix_file_name = 'task1_subject_weight_matrix.csv', 
                            type_weight_matrix_file_name = 'task2_type_weight_matrix.csv'):
    latent_semantics = right_factor_matrix
    store_array_as_csv(latent_semantics, output_folder, latent_semantics_file_name)

    if filter == "type":
        subject_weight_matrix = np.array(get_subject_weight_matrix(images_with_attributes, left_factor_matrix))
        store_array_as_csv(subject_weight_matrix, output_folder, subject_weight_matrix_file_name)        
        sort_print_matrix(subject_weight_matrix, k=len(subject_weight_matrix[0]))
    elif filter == "subject_id":
        type_weight_matrix = np.array(get_type_weight_matrix(images_with_attributes, left_factor_matrix))
        store_array_as_csv(type_weight_matrix, output_folder, type_weight_matrix_file_name)   
        sort_print_matrix(type_weight_matrix, len(type_weight_matrix[0]), "t")


# Print the sorted subject weight or task weight based on the inputs called in the above method.
def sort_print_matrix(subject_weight_matrix, k, one="s"):
    sorted_weights_for_each_latent_semantic = {}
    subject_weight_matrix = np.array(subject_weight_matrix).transpose()
    for i, val in enumerate(subject_weight_matrix):
        latent_sematic = f"l{i}"
        subject_weight_sort = {}
        for j, v in enumerate(val):
            subject = f"{one}{j}"
            subject_weight_sort[subject] = v
        subject_weight_sort = sorted(subject_weight_sort.items(), key=lambda kv: kv[1], reverse=True)
        sorted_weights_for_each_latent_semantic[latent_sematic] = subject_weight_sort
        print(f"{latent_sematic} - {sorted_weights_for_each_latent_semantic[latent_sematic]}")