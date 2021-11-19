from input_output import get_images_with_attributes, get_image_arr_from_dict, store_array_as_csv, get_images_and_attributes_from_folder, get_label_arr_from_dict
from .features import *
from .pca import compute_pca
from .svd import compute_svd
from .kmeans import kmeans
from .lda import LDA
from .dim_red import perform_dim_red

from .weight_matrices_calc import get_subject_weight_matrix, get_type_weight_matrix

def supply_inputs_to_ppr(input_folder_path = 'Dataset', feature_model = 'elbp', k = 100, label_name = 'type', 
	dim_red_technique = 'svd', output_folder='output',
    latent_semantics_file_name='task1_latent_semantics.csv'):
	images_with_attributes = get_images_and_attributes_from_folder(input_folder_path)
	images = get_image_arr_from_dict(images_with_attributes)
	labels = get_label_arr_from_dict(images_with_attributes, label_name = label_name)
	image_features = get_flattened_features_for_images(images, feature_model)

	dim_red_technique = dim_red_technique.lower()
	left_factor_matrix, right_factor_matrix = perform_dim_red(dim_red_technique, image_features, k)
	print(left_factor_matrix)


supply_inputs_to_ppr(input_folder_path = 'Dataset', feature_model = 'elbp', k = 100, label_name = 'type', 
	dim_red_technique = 'svd')