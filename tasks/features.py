from .Feature_models.ColorMoments import *
from .Feature_models.elbp import *
from .Feature_models.hog import *
from sklearn import preprocessing


# Will flatten the images
def get_flattened_features_for_images(images, feature_model):
	print(f'Getting {feature_model} features.')
	flattened_features_list = []
	for image in images:
		image_features = compute_features(image, feature_model)
		#print(f"{len(image_features)}, feature:{feature_model} - Get flattened features.")
		flattened_features_list.append(image_features.flatten())
		#print(f"{len(flattened_features_list[-1])}, feature:{feature_model} - Get flattened features size after flattened.")
	return flattened_features_list


# compute the features and return the values
def compute_features(image, feature_model):
	feature_model = feature_model.lower()
	if feature_model == 'elbp':
		return elbp(image)
	elif feature_model == 'cm':
		return color_moments(image)
	elif feature_model == 'hog':
		return histogram_of_gradients(image)


# Flatten the elbp image data
def elbp(image):
	return compute_elbp(image).flatten()


# Combine the color moments
def color_moments(image):
	features = compute_color_moment_of_image(image)
	features = preprocessing.normalize(features)
	features[0] = (features[0] + features[1] + features[2]) / 3.0
	return features[0]


# Compute the hog.
def histogram_of_gradients(image):
	return computer_hog(image)
