from .cache import *
from .lsh import LocalitySensitiveHashing


def task4(input_folder, feature_model, num_layers, num_hashes_per_layer, query_image_path, num_similar_images_to_retrieve):
    images_with_attributes, image_features  = get_images_with_attributes_and_features(input_folder, feature_model)
    lsh_obj = LocalitySensitiveHashing(num_layers, num_hashes_per_layer, image_features)

    hash_buckets_per_layer = lsh_obj.get_hash_buckets_per_layer()
    for hash_buckets_dict in hash_buckets_per_layer:
        for key in hash_buckets_dict:
            print(key, len(hash_buckets_dict[key]))




