import numpy as np

class LocalitySensitiveHashing:

    def __init__(self, num_layers, num_hashes_per_layer, input_vectors):
        self.num_layers = num_layers
        self.num_hashes_per_layer = num_hashes_per_layer
        self.input_vectors = input_vectors
        number_of_features = len(input_vectors[0])
        self.random_planes = [np.random.randn(self.num_hashes_per_layer, number_of_features)
                              for i in range(self.num_layers)]
        self.layer_buckets = [{} for i in range(self.num_layers)]
        self.create_index_structure_with_input_vectors()

    def create_index_structure_with_input_vectors(self):
        input_vectors = self.input_vectors
        layer_buckets = self.layer_buckets
        for input_vector in input_vectors:
            hash_codes = self.get_hash_codes_for_object(input_vector)
            for i in range(self.num_layers):
                buckets_dict = layer_buckets[i]
                hash_code = hash_codes[i]
                if hash_code not in buckets_dict:
                    buckets_dict[hash_code] = []
                buckets_dict[hash_code].append(input_vector)

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

    def get_layer_buckets(self):
        return self.layer_buckets