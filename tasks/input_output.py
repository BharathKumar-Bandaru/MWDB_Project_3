import os
from skimage.io import imread
import numpy as np
import pandas as pd
import matplotlib as mpl
from .image import Image

#dictionary of image datasets based on folder
#folder_name is the key, list of image_dict is the value
image_dataset_dict = {}


# Get the images and store them in dictionary
def get_images_and_attributes_from_folder(folder_path):
    """
    retrieves all images and stores in a dictionary
    folder_path: relative path to images dataset folder
    """
    if folder_path in image_dataset_dict:
        return image_dataset_dict[folder_path]

    images_with_attributes = [] #list of dictionaries
    for entry in os.scandir(folder_path):
        if entry.path.endswith('.png') and entry.is_file():
            filename = entry.name
            image = imread(entry.path, as_gray = True)
            image = np.array(image)
            file_attributes = filename.split('-')
            image_dict = {
                "filename": filename,
                "type": file_attributes[1],
                "subject_id": file_attributes[2],
                "image_id": file_attributes[3].split('.')[0],
                "image": image
            }
            images_with_attributes.append(image_dict)

    image_dataset_dict[folder_path] = images_with_attributes
    return images_with_attributes

def get_image_objects_from_folder(folder_path):
    image_objects = []
    for entry in os.scandir(folder_path):
        if entry.path.endswith('.png') and entry.is_file():
            filename = entry.name
            image = imread(entry.path, as_gray = True)
            image_obj = Image(filename = filename, image_arr = image)
            image_objects.append(image_obj)
    return image_objects

# Filter the images based on search criteria.
def filter_images(images_with_attributes, filter_based_on = 'none', filter_value = ''):
    """
    images_with_attributes: images with attributes
    filter_based_on: 'none' | 'type' | 'subject_id'
    filter_value: value for the filter

    """
    print(f"Filtering images based on {filter_based_on}:{filter_value}")
    filtered_images_with_attr = []
    if(filter_based_on == 'none'):
        return images_with_attributes
    
    for image in images_with_attributes:
        if image[filter_based_on] == filter_value:
            filtered_images_with_attr.append(image)

    return filtered_images_with_attr


# Function call to get the images with all basic attributes
def get_images_with_attributes(folder_path, filter, filter_value):
    print(f'Getting images and attributes from {folder_path}')
    images_with_attributes = get_images_and_attributes_from_folder(folder_path)
    filtered_images_with_attributes = filter_images(images_with_attributes, filter, filter_value)
    return filtered_images_with_attributes


# Get the image data from the dict that we calculated
def get_image_arr_from_dict(images_with_attributes):
    """
    retrieves a list of images from dictionary (using the key 'image')
    """
    return [image_dict['image'] for image_dict in images_with_attributes]

def get_label_arr_from_dict(images_with_attributes, label_name):
    """
    label_name: 'type', 'subject_id', or 'image_id'
    """
    return [image_dict[label_name] for image_dict in images_with_attributes]

def get_image_objects_from_dict(images_with_attributes, features_list):
	image_objects = []
	for i in range(len(images_with_attributes)):
		image_dict = images_with_attributes[i]
		image_obj = Image(filename = image_dict['filename'], image_arr = image_dict['image'], type = image_dict['type'],
						  subject_id = image_dict['subject_id'], image_id = image_dict['image_id'],
                          features = features_list[i])
		image_objects.append(image_obj)
	return image_objects




# Store the values to a csv file.
def store_array_as_csv(array, folder_path, file_name):
    print(f'Saving {file_name}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    array = np.array(array)
    df = pd.DataFrame(array)
    file_path = os.path.join(folder_path, file_name)
    df.to_csv(file_path, index = False, header = False)


# Read the image from the dataset
def get_image_arr_from_file(file_path):
    image = imread(file_path, as_gray = True)
    return image


# Save the image
def save_image(imageArr, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    mpl.image.imsave(fname = os.path.join(folder_path, file_name), arr = imageArr, cmap = 'gray')


# Clear the unwanted contents
def clear_folder_contents(folder_path):
    if os.path.isdir(folder_path):
        for entry in os.scandir(folder_path):
            os.remove(entry)


# Save the images.
def save_images_by_clearing_folder(image_file_name_tuple_list, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        clear_folder_contents(folder_path)
    for imageArr, file_name in image_file_name_tuple_list:
        mpl.image.imsave(fname = os.path.join(folder_path, file_name), arr = imageArr, cmap = 'gray')

# Mapping class for task 1
class_map = {
    'cc' : 1,
    'con' : 2,
    'emboss': 3,
    'jitter': 4,
    'neg': 5,
    'noise01': 6,
    'noise02': 7,
    'original': 8,
    'poster': 9,
    'rot': 10,
    'smooth': 11,
    'stipple': 12
}

def process_labels(images_data, filter_type = 'type'):
    labels = []
    for each in images_data:
        label = class_map[each['type']] if filter_type == "type" else int(each[filter_type])
        labels.append(label)
    return labels