from tasks.input_output import *
from tasks.features import *
from tasks.dim_red import perform_dim_red
import numpy as np
from numpy import genfromtxt
from tasks.PPRClassifier import PersonalizedPageRankClassifier

def get_inputs_for_ppr(latent_semantics, images_with_attributes, image_features,
                       test_images_with_attributes, test_image_features):
    #latent_semantics = np.matrix(genfromtxt(latent_semantics_file_path, delimiter=','))
    transposed_latent_semantics = np.matrix.transpose(latent_semantics)
    image_objects = get_image_objects_from_dict(images_with_attributes, image_features)
    for i in range(len(image_objects)):
        image_obj = image_objects[i]
        latent_features = np.matmul(image_obj.features, transposed_latent_semantics)
        image_obj.set_latent_features(latent_features)

    test_image_objects = get_image_objects_from_dict(test_images_with_attributes, test_image_features)
    for i in range(len(test_image_objects)):
        test_image_obj = test_image_objects[i]
        latent_features = np.matmul(test_image_obj.features, transposed_latent_semantics)
        test_image_obj.set_latent_features(latent_features)
    return image_objects, test_image_objects

def run_ppr(image_objects, test_image_objects, classification_label = 'type', num_nodes_to_consider_for_classifying = 15, num_similar_nodes_to_form_graph = 15):
    ppr_classifier = PersonalizedPageRankClassifier(input_image_objects = image_objects,
                                         test_image_objects = test_image_objects,
                                         classification_label = classification_label,
                                        num_nodes_to_consider_for_classifying = num_nodes_to_consider_for_classifying,
                                        num_similar_nodes_to_form_graph = num_similar_nodes_to_form_graph)
    labels = ppr_classifier.get_classified_labels()
    #print('\n-------' + classification_label + ' labels: --------')
    #print(labels)
    #print('\n-----------------------------------------------------------')
    return labels


def perform_ppr_classification(latent_semantics, images_with_attributes, image_features,
                                test_images_with_attributes, test_image_features, label_name):
    image_objects, test_image_objects = get_inputs_for_ppr(latent_semantics, images_with_attributes,
                                                           image_features, test_images_with_attributes,
                                                           test_image_features)
    if label_name == 'type':
        num_nodes_to_consider_for_classifying = 1
        num_similar_nodes_to_form_graph = 15
    elif label_name == 'subject_id':
        num_nodes_to_consider_for_classifying = 1
        num_similar_nodes_to_form_graph = 15
    else:
        num_nodes_to_consider_for_classifying = 1
        num_similar_nodes_to_form_graph = 15

    correct_labels = []
    for i in range(len(test_image_objects)):
        correct_labels.append(getattr(test_image_objects[i], label_name))
    predicted_labels = run_ppr(image_objects, test_image_objects, label_name, num_nodes_to_consider_for_classifying, num_similar_nodes_to_form_graph)

    return predicted_labels, correct_labels