import os

from tasks.lsh import *

task4(input_folder='Dataset_100', feature_model='elbp', num_layers=5, num_hashes_per_layer=4,
      query_image_path=os.path.join('Test_Dataset','image-cc-1-1.png'), num_similar_images_to_retrieve=2)