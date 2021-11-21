from classifiers import decision_tree_2, decision_tree
from csv import reader
import numpy as np
from skimage.io import imread


def readImage(filename):
    image = imread(filename, as_gray=True)
    return np.array(image)

def load_csv(filename):
    file = open(filename, 'rt')
    lines = reader(file)
    # convert str -> float

    dataset = [list(map(float, row)) for row in lines]
    print("here: " + str(len(dataset)))
    val = np.zeros(13)
    for row in dataset:
        val[int(row[5])]+=1

    print("printing label counts ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for i in range(len(val)):
        print(str(i) + "__" + str(val[i]))
    return dataset

#decision_tree.read_dataset_and_construct_tree()
dc =  decision_tree_2.DecisionTreeClassifier()
rows = load_csv("/Users/arushigaur/Documents/masters_projects/mwdb/Mwdb_Project_2/output/left_factor_matrix.csv")
print("dataset read")
root = dc.make_tree(rows)
print("tree formed")
image = readImage("Dataset/image-rot-28-4.png")
print("predictions: ")
print(dc.predict(image[0], root))
print(type(root))
dc.print_tree(root)