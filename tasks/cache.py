cache_for_input_images = {} #{input_folder_path}_{feature_model}_{dim_red_technique}_{k}, {'images_with_attributes': , 'images':, 'image_features':, 'latent_semantics_file_path': }
cache_for_folder_image_features = {} #input_folder_path: {'images': , 'image_features':}

def get_cache_for_input_images():
    return cache_for_input_images

def get_cache_for_folder_image_features():
    return cache_for_folder_image_features


