import os

from tasks.lsh import *
from tasks.task4 import task4
import icecream

result = task4(input_folder='Dataset', feature_model='elbp', num_layers=3, num_hashes_per_layer=10,
      query_image_path=os.path.join('Dataset','image-con-35-3.png'), num_similar_images_to_retrieve=5)

for each in result:
      icecream.ic(each.features)