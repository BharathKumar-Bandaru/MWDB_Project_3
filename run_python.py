from classifiers import decision_tree_2, decision_tree
from csv import reader
import numpy as np
from skimage.io import imread
from tasks.features import compute_features
from tasks.input_output import calculate_latent_semantics_with_type_labels
from tasks.input_output import get_images_and_attributes_from_folder, get_image_arr_from_dict, get_type_label_arr_from_dict, get_label_arr_from_dict


def readImage(filename):
    image = imread(filename, as_gray=True)
    return np.array(image)


def load_csv(filename):
    file = open(filename, 'rt')
    lines = reader(file)
    # convert str -> float

    dataset = [list(map(float, row)) for row in lines]
    print("here: " + str(len(dataset)))
    val = np.zeros(41)
    for row in dataset:
        val[int(row[5])] += 1

    # print("printing label counts ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # for i in range(len(val)):
    #     print(str(i) + "__" + str(val[i]))
    return dataset


#calculate_latent_semantics_with_type_labels("hog", 5, "pca", "type", "1000", "output")

#decision_tree.read_dataset_and_construct_tree()
dc =  decision_tree_2.DecisionTreeClassifier()
rows = load_csv("/Users/arushigaur/Documents/masters_projects/mwdb/Mwdb_Project_3/output/left_factor_matrix_type.csv")
images_attr = get_images_and_attributes_from_folder("100")
images  = get_image_arr_from_dict(images_attr)
labels = get_type_label_arr_from_dict(images_attr)
#labels = get_label_arr_from_dict(images_attr, "subject_id")

latent_semantics = load_csv("/Users/arushigaur/Documents/masters_projects/mwdb/Mwdb_Project_3/output/right_factor_matrix_type.csv")
print("dataset read")
root = dc.make_tree(rows)
# print("tree formed")
# image = readImage("Dataset/image-jitter-23-5.png")
# image_features = compute_features(image, "hog")
features = []
for image in images:
    features.append(compute_features(image, "hog"))
print("length:")
print(str(len(features)))
print(str(len(latent_semantics)) + " " + str(len(latent_semantics[0])))

image_new_space = np.matmul(features, np.transpose(latent_semantics))
print("prediction: ")
print(str(len(image_new_space)))
count = 0
totalCount = 0
for i in range(len(image_new_space)):
    prediction = dc.predict(image_new_space[i], root)
    if int(prediction) == int(labels[i]):
        count+=1
    totalCount+=1
print(count/totalCount)

# print(type(root))
# dc.print_tree(root)
