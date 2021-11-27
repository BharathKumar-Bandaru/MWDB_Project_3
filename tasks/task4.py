from .cache import *
from .lsh import LocalitySensitiveHashing
from .input_output import *

def task4(input_folder, feature_model, num_layers, num_hashes_per_layer, query_image_path, num_similar_images_to_retrieve,
          output_folder_path = 'lsh_output'):
    images_with_attributes, image_features  = get_images_with_attributes_and_features(input_folder, feature_model)
    image_objects = get_image_objects_from_dict(images_with_attributes, image_features)
    print('Initializing Locality Sensitive Hashing index structure')
    lsh_obj = LocalitySensitiveHashing(num_layers, num_hashes_per_layer, image_objects)

    query_image_obj = get_image_object_from_file(query_image_path)
    query_image_obj.set_features(get_flattened_features_for_a_single_image(query_image_obj.image_arr, feature_model))

    result_images = lsh_obj.get_similar_objects(query_image_obj, num_similar_images_to_retrieve)
    image_file_name_tuple_list = [(image_obj.image_arr, image_obj.filename) for image_obj in result_images]
    save_images_by_clearing_folder(image_file_name_tuple_list, output_folder_path)
    print(f'LSH result images are saved in {output_folder_path}')

    ##Testing##
    hash_buckets_per_layer = lsh_obj.get_hash_buckets_per_layer()
    for i in range(len(hash_buckets_per_layer)):
        print('-----------')
        print(f'Layer {i+1}')
        hash_buckets_dict = hash_buckets_per_layer[i]
        for key in hash_buckets_dict:
            print(key, len(hash_buckets_dict[key]))




