import numpy as np

from tasks.dim_red import perform_dim_red
from tasks.features import get_flattened_features_for_images
from tasks.input_output import get_images_with_attributes

class_map = {
    'cc' : 1,
    'con' : 2,
    'emboss': 3,
    'jitter': 4,
    'neg': 5,
    'noise01': 6,
    'noise02': 7,
    'original': 8,
    'poster': 9,
    'rot': 10,
    'smooth': 11,
    'stipple': 12
}

def process_data(images_data, filter_type = 'type'):
    data_values = []
    labels = []
    for each in images_data:
        data_values.append(each['image'])
        label = class_map[each['type']] if filter_type == "type" else each[filter_type]
        labels.append(label)
    return data_values, labels

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def retrive_features_task1(folder_path, feature_model, dim_red, k, filter="type"):
    # Data retrieval for given folder
    images_data = get_images_with_attributes(folder_path=folder_path, filter='none', filter_value=None)
    X_data, labels = process_data(images_data, filter_type=filter)
    # Do the feature extraction
    feature_model = feature_model
    features = get_flattened_features_for_images(X_data, feature_model)
    # Do the dimentionality reduction
    k = k
    new_features = perform_dim_red(dim_red, features, k)
    return np.array(new_features[1]), np.array(labels), np.array(features)