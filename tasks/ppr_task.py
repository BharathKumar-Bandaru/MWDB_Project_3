


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

    predicted_labels = run_ppr(image_objects, test_image_objects, label_name, num_nodes_to_consider_for_classifying, num_similar_nodes_to_form_graph)
    correct_labels = None

    return predicted_labels, correct_labels

def get_inputs_for_ppr(latent_semantics, images_with_attributes, image_features,
                       test_images_with_attributes, test_image_features):
    """
        Fill this
    """
    return image_objects, test_image_objects