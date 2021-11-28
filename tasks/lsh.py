import numpy as np
from scipy.spatial import distance
import sys

class LocalitySensitiveHashing:

    def __init__(self, num_layers, num_hashes_per_layer, input_image_objects):
        self.num_layers = num_layers
        self.num_hashes_per_layer = num_hashes_per_layer
        self.input_image_objects = input_image_objects
        number_of_features = len(input_image_objects[0].features)
        self.random_planes = [np.random.randn(self.num_hashes_per_layer, number_of_features)
                              for i in range(self.num_layers)]
        self.hash_buckets_per_layer = [{} for i in range(self.num_layers)]
        self.create_index_structure_with_input_vectors()
        self.num_buckets_searched = 0
        self.latest_query_image_obj = None
        self.latest_query_results = None

    def create_index_structure_with_input_vectors(self):
        hash_buckets_per_layer = self.hash_buckets_per_layer
        for idx, image_obj in enumerate(self.input_image_objects):
            input_vector = image_obj.features
            hash_codes = self.get_hash_codes_for_object(input_vector)
            for i in range(self.num_layers):
                buckets_dict = hash_buckets_per_layer[i]
                hash_code = hash_codes[i]
                if hash_code not in buckets_dict:
                    buckets_dict[hash_code] = []
                buckets_dict[hash_code].append(idx)

    def get_hash_codes_for_object(self, input_vector):
        hash_codes = []
        for layer_no in range(self.num_layers):
            hash_code = ""
            for plane in self.random_planes[layer_no]:
                dot_product = input_vector.dot(plane)
                if dot_product < 0:
                    hash_code += '0'
                else:
                    hash_code += '1'
            hash_codes.append(hash_code)
        return hash_codes

    def retrieve_objects_in_bucket(self, layer_num, hashcode):
        object_indices = self.hash_buckets_per_layer[layer_num][hashcode]
        return [self.input_image_objects[index] for index in object_indices]

    def get_hash_buckets_per_layer(self):
        return self.hash_buckets_per_layer

    def compute_distance(self, vector1, vector2):
        return distance.euclidean(vector1, vector2)

    def get_image_indices_with_distances_sorted(self, query_image_obj, image_object_list):
        result = []
        for idx, image in enumerate(image_object_list):
            dist = self.compute_distance(query_image_obj.features, image.features)
            result.append((idx, dist))
        result.sort(key=lambda x: x[1])
        return result

    def get_similar_objects(self, query_image_obj, num_similar_images_to_retrieve):
        self.num_buckets_searched = 0
        self.overall_images_considered = 0
        self.unique_images_considered = 0

        hash_codes = self.get_hash_codes_for_object(query_image_obj.features)
        print(f'Hash codes of the query image in different layers: {hash_codes}')
        num_images_in_retrieved_buckets = 0
        objects_retrieved_in_diff_layers = [] #list of sets
        for idx, hash_code in enumerate(hash_codes):
            images = self.retrieve_objects_in_bucket(idx, hash_code)
            num_images_in_retrieved_buckets += len(images)
            objects_retrieved_in_diff_layers.append(set(images))
            self.num_buckets_searched += 1

        bucket_intersection_set = set.intersection(*objects_retrieved_in_diff_layers)
        bucket_union_set = set.union(*objects_retrieved_in_diff_layers)


        image_object_list = list(bucket_union_set)
        self.overall_images_considered = num_images_in_retrieved_buckets

        """
        image_object_list = []
        if len(bucket_intersection_set) >= num_similar_images_to_retrieve:
            image_object_list = list(bucket_intersection_set)
            self.overall_images_considered = len(bucket_intersection_set) * self.num_layers #elements in intersection set are present in all sets (no. of hash codes of query object = no. of layers)
        else:
            image_object_list = list(bucket_union_set)
            self.overall_images_considered = num_images_in_retrieved_buckets
        """
        self.unique_images_considered = len(image_object_list)

        image_indices_and_dist = self.get_image_indices_with_distances_sorted(query_image_obj, image_object_list)
        result_image_objects = []
        n = min(num_similar_images_to_retrieve, len(image_indices_and_dist))  #returning only the available images in hash buckets
        for i in range(n):
            image_obj_index = image_indices_and_dist[i][0]
            result_image_objects.append(image_object_list[image_obj_index])

        self.latest_query_image_obj = query_image_obj
        self.latest_query_results = list(result_image_objects)
        return result_image_objects

    def compute_and_print_false_positives_and_misses(self):
        if self.latest_query_image_obj is None or self.latest_query_results is None:
            print('No Query image object or query results found')
            return
        retrieved_results = []
        for image_obj in self.latest_query_results:
            retrieved_results.append(image_obj.filename)

        n = len(self.latest_query_results)
        image_indices_and_dist_tuples = self.get_image_indices_with_distances_sorted(self.latest_query_image_obj, self.input_image_objects)

        correct_results = []
        for i in range(n):
            image_obj_index = image_indices_and_dist_tuples[i][0]
            correct_results.append(self.input_image_objects[image_obj_index].filename)

        retrieved_results = set(retrieved_results)
        correct_results = set(correct_results)

        miss_rate = len(correct_results - retrieved_results)/n #false negative rate
        false_positives = len(retrieved_results - correct_results)

        print(f'False Positives: {false_positives}')
        print(f'Miss rate: {miss_rate}')

    def print_index_structure_stats(self):
        print('\n--------------')
        print('LSH Index Structure Stats:')
        hash_buckets_per_layer = self.hash_buckets_per_layer

        """
        print('No. of objects per hash code in each layer:')
        for i in range(len(hash_buckets_per_layer)):
            print(f'Layer {i + 1}:')
            hash_buckets_dict = hash_buckets_per_layer[i]
            for key in hash_buckets_dict:
                print(f'{key}: {len(hash_buckets_dict[key])}')
            print('-----------')
        """
        print(f'Total index structure size in bytes: {sys.getsizeof(hash_buckets_per_layer)}')
        print(f'Number of buckets searched: {self.num_buckets_searched}')
        print(f'Number of input images: {len(self.input_image_objects)}')
        print(f'Number of overall images considered (with overlaps): {self.overall_images_considered}')
        print(f'Number of unique images considered: {self.unique_images_considered}')

        print('Computing false positives and miss rates...')
        self.compute_and_print_false_positives_and_misses()
