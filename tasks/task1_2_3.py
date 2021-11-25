from .input_output import *
from .features import *
from .dim_red import perform_dim_red
from numpy import genfromtxt

from .weight_matrices_calc import get_subject_weight_matrix, get_type_weight_matrix
from .pca import compute_pca
from .svd import compute_svd
from .kmeans import kmeans
from .lda import LDA


cache_for_input_images = {} #input_folder_path, {'images_with_attributes': , 'images':, 'image_features':, 'latent_semantics_file_path': }
latent_semantics_file_id = 0

# Entry for tasks 1,2, and 3
def task_1_2_3(task_number, input_folder_path, feature_model, k, test_folder_path, classifier, label_type,
               dim_red_technique = 'svd', output_folder = 'output', latent_semantics_file_name = None,
               use_cached_input_images = True):

    if use_cached_input_images is True and input_folder_path in cache_for_input_images:
        images_with_attributes = cache_for_input_images[input_folder_path]['images_with_attributes']
        image_features = cache_for_input_images[input_folder_path]['image_features']
        latent_semantics_file_path = cache_for_input_images[input_folder_path]['latent_semantics_file_path']
        latent_semantics = np.matrix(genfromtxt(latent_semantics_file_path, delimiter=','))
    else:
        images_with_attributes = get_images_and_attributes_from_folder(input_folder_path)
        images = get_image_arr_from_dict(images_with_attributes)
        image_features = get_flattened_features_for_images(images, feature_model)
        dim_red_technique = dim_red_technique.lower()
        left_factor_matrix, right_factor_matrix = perform_dim_red(dim_red_technique, image_features, k)
        latent_semantics = right_factor_matrix

        if latent_semantics_file_name is None:
            latent_semantics_file_id += 1
            latent_semantics_file_name = f'[{latent_semantics_file_id}]_task_{task_number}_{feature_model}_{dim_red_technique}_{k}_latent_semantics.csv'
        store_array_as_csv(latent_semantics, output_folder, latent_semantics_file_name)

        #Storing into cache
        cache_for_input_images[input_folder_path] = {'images_with_attributes': images_with_attributes,
                                                     'images': images,
                                                     'image_features': image_features,
                                                     'latent_semantics_file_path': os.path.join(output_folder, latent_semantics_file_name)}

    test_images_with_attributes = get_images_and_attributes_from_folder(test_folder_path)
    test_images = get_image_arr_from_dict(test_images_with_attributes)
    test_image_features = get_flattened_features_for_images(test_images, feature_model)

    classifier = classifier.lower()
    if classifier == 'decision-tree':
        perform_decision_tree_classification(latent_semantics, images_with_attributes, image_features,
                                             test_images_with_attributes, test_image_features)
    elif classifier == 'svm':
        perform_svm_classification(latent_semantics, images_with_attributes, image_features,
                                             test_images_with_attributes, test_image_features)
    elif classifier == 'ppr':
        perform_ppr_classification(latent_semantics, images_with_attributes, image_features,
                                             test_images_with_attributes, test_image_features)

def perform_decision_tree_classification(latent_semantics, images_with_attributes, image_features,
                                             test_images_with_attributes, test_image_features):
    print('Decision Tree Classifier')

def perform_svm_classification(latent_semantics, images_with_attributes, image_features,
                                         test_images_with_attributes, test_image_features):
    print('Support Vector Machine Classifier')

def perform_ppr_classification(latent_semantics, images_with_attributes, image_features,
                                         test_images_with_attributes, test_image_features):
    print('Personalized Page Rank Classifier')

"""
def task1_2_3(feature_model, filter, image_type, k, dim_red_technique,
            folder_path='input_images', output_folder='output',
            latent_semantics_file_name='task1_latent_semantics.csv'):

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

"""