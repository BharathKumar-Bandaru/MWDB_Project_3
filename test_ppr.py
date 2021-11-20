from tasks.input_output import *
from tasks.features import *
from tasks.dim_red import perform_dim_red
import numpy as np
from tasks.ppr import PersonalizedPageRank

def supply_inputs_to_ppr(input_folder_path = 'Dataset', test_folder_path = 'Dataset', feature_model = 'elbp', k = 100,
						 label_name = 'type', dim_red_technique = 'svd', output_folder='output',
						 latent_semantics_file_name='task1_latent_semantics.csv'):
	images_with_attributes = get_images_and_attributes_from_folder(input_folder_path)
	images = get_image_arr_from_dict(images_with_attributes)
	#labels = get_label_arr_from_dict(images_with_attributes, label_name = label_name)
	image_features = get_flattened_features_for_images(images, feature_model)

	dim_red_technique = dim_red_technique.lower()
	left_factor_matrix, right_factor_matrix = perform_dim_red(dim_red_technique, image_features, k)

	image_objects = get_image_objects_from_dict(images_with_attributes)
	for i in range(len(image_objects)):
		image_obj = image_objects[i]
		img_latent_features = left_factor_matrix[i]
		image_obj.set_latent_features(img_latent_features)

	test_image_objects = get_image_objects_from_folder(test_folder_path)

#	assign features to test_image_objects -
	for image_object in test_image_objects:
		image = image_object.image_arr
		features = get_flattened_features_for_a_single_image(image,feature_model)
		latent_features = np.matmul(features, np.matrix.transpose(right_factor_matrix))
		image_object.set_latent_features(latent_features)

	test(image_objects, test_image_objects)

def test(image_objects, test_image_objects):
	ppr_class = PersonalizedPageRank(input_image_objects = image_objects, test_image_object = test_image_objects[0])
	ppr_class.get_classified_label()


supply_inputs_to_ppr(input_folder_path='Sample_Dataset', test_folder_path='Sample_Dataset', feature_model='elbp', k=100,
						 label_name='type',
						 dim_red_technique='svd')

