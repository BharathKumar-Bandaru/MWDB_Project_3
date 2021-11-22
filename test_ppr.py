from tasks.input_output import *
from tasks.features import *
from tasks.dim_red import perform_dim_red
import numpy as np
from numpy import genfromtxt
from tasks.PPRClassifier import PersonalizedPageRankClassifier

def task(input_folder_path = 'Dataset', test_folder_path = 'Dataset', feature_model = 'elbp', k = 100,
						 label_name = 'type', dim_red_technique = 'svd', output_folder='output',
						 latent_semantics_file_name='task1_latent_semantics.csv'):
	images_with_attributes = get_images_and_attributes_from_folder(input_folder_path)
	images = get_image_arr_from_dict(images_with_attributes)
	image_features = get_flattened_features_for_images(images, feature_model)

	#dim_red_technique = dim_red_technique.lower()
	#left_factor_matrix, right_factor_matrix = perform_dim_red(dim_red_technique, image_features, k)

	#latent_semantics = right_factor_matrix
	#store_array_as_csv(latent_semantics, output_folder, latent_semantics_file_name)
	process_ppr(latent_semantics_file_path = os.path.join(output_folder, latent_semantics_file_name), images_with_attributes = images_with_attributes, image_features = image_features, test_folder_path = test_folder_path, feature_model = feature_model)

def process_ppr(latent_semantics_file_path, images_with_attributes, image_features, test_folder_path, feature_model):
	image_objects, test_image_objects = get_inputs_for_ppr(latent_semantics_file_path, images_with_attributes, image_features, test_folder_path, feature_model)
	run_ppr(image_objects, test_image_objects, 'type', 3, 15)
	run_ppr(image_objects, test_image_objects, 'subject_id', 3, 45)
	run_ppr(image_objects, test_image_objects, 'image_id', 3, 12)

def get_inputs_for_ppr(latent_semantics_file_path, images_with_attributes, image_features, test_folder_path, feature_model):
	latent_semantics = np.matrix(genfromtxt(latent_semantics_file_path, delimiter=','))
	transposed_latent_semantics = np.matrix.transpose(latent_semantics)
	image_objects = get_image_objects_from_dict(images_with_attributes, image_features)
	for i in range(len(image_objects)):
		image_obj = image_objects[i]
		latent_features = np.matmul(image_obj.features, transposed_latent_semantics)
		image_obj.set_latent_features(latent_features)

	test_image_objects = get_image_objects_from_folder(test_folder_path)

	for image_object in test_image_objects:
		image = image_object.image_arr
		features = get_flattened_features_for_a_single_image(image,feature_model)
		latent_features = np.matmul(features, transposed_latent_semantics)
		image_object.set_latent_features(latent_features)

	return image_objects, test_image_objects

def run_ppr(image_objects, test_image_objects, classification_label = 'type', num_nodes_to_consider_for_classifying = 15, num_similar_nodes_to_form_graph = 15):
	ppr_classifier = PersonalizedPageRankClassifier(input_image_objects = image_objects,
										 test_image_objects = test_image_objects,
										 classification_label = classification_label,
										num_nodes_to_consider_for_classifying = num_nodes_to_consider_for_classifying,
										num_similar_nodes_to_form_graph = num_similar_nodes_to_form_graph)
	labels = ppr_classifier.get_classified_labels()
	print('\n-------' + classification_label + ' labels: --------')
	print(labels)
	print('\n-----------------------------------------------------------')

task(input_folder_path='Dataset_100', test_folder_path='Test_Dataset', feature_model='elbp', k=100,
						 label_name='type',
						 dim_red_technique='svd')