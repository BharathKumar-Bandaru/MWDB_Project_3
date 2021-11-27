import os

from tasks.lsh import *
from tasks.task4 import task4
import icecream
from tasks.task1_2_3 import task_1_2_3

# result = task4(input_folder='Dataset', feature_model='elbp', num_layers=3, num_hashes_per_layer=10,
#       query_image_path=os.path.join('Dataset','image-con-35-3.png'), num_similar_images_to_retrieve=5)

# for each in result:
#       icecream.ic(each.features)

task_1_2_3(task_number = 1, input_folder_path = '1000', feature_model = "elbp", k =5, test_folder_path = '100', classifier =  'decision-tree',
               dim_red_technique = 'pca', output_folder = 'output', latent_semantics_file_name = None,
               use_cached_input_images = True)