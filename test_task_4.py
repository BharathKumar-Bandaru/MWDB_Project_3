import os

from tasks.lsh import *
from tasks.task4 import task4

task4(input_folder='Dataset', feature_model='elbp', num_layers=5, num_hashes_per_layer=3,
      query_image_path=os.path.join('Dataset','image-con-35-3.png'), num_similar_images_to_retrieve=10)