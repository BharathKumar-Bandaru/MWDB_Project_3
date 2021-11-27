from .input_output import *
from .features import *

cache_for_input_images = {} #{input_folder_path}_{feature_model}_{dim_red_technique}_{k}, {'images_with_attributes': , 'images':, 'image_features':, 'latent_semantics_file_path': }

cache_for_image_attributes_and_features = {}

def get_cache_for_input_images():
    return cache_for_input_images

def get_images_with_attributes_and_features(folder_path, feature_model):
    if (folder_path, feature_model) not in cache_for_image_attributes_and_features:
        images_with_attributes = get_images_and_attributes_from_folder(folder_path)
        images = get_image_arr_from_dict(images_with_attributes)
        image_features = get_flattened_features_for_images(images, feature_model)
        cache_for_image_attributes_and_features[(folder_path, feature_model)] = \
            {'images_with_attributes': images_with_attributes,
             'image_features': image_features}
    return cache_for_image_attributes_and_features[folder_path, feature_model]['images_with_attributes'], \
            cache_for_image_attributes_and_features[folder_path, feature_model]['image_features']


