import os

import numpy as np
from skimage.io import imread
from tasks.input_output import *
import warnings
from tasks.task1_2_3 import *
from tasks.task8 import *

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
np.set_printoptions(linewidth=np.inf)
warnings.filterwarnings("ignore")

task_type_map = {1:"type", 2:"subject_id", 3:"image_id"}

if __name__ == "__main__":
    dataset_path = "Dataset"
    print("Enter the task number, to exit type e: ")
    task_number = input()


    while task_number.isnumeric():
        task_number = int(task_number)

        if task_number <= 3:

            print("Enter the dataset path:")
            data_path = input()
            print("Enter the test images path:")
            test_path = input()
            print("\nFeatures: [cm, elbp, hog]")
            print("Enter the feature from the above list:")
            f = input()
            print(f"Enter the value of k:")
            print(f"To use all the features with out dimensionality reduction enter 'all'.")
            k = input()
            k = int(k) if k != "all" else k
            while f == "cm" and type(k) == int and k > 63:
                print(f"For entered value of feature:{f} the value of k:{k} should be less than 64.")
                print(f"Re-enter the value of k:")
                k = input()
                k = int(k) if k != "all" else k

            print(f"Enter the classifier [svm, decision-tree, ppr]:")
            classifier = input()

            dr = 'pca'
            print(f"The dimentionality reduction being used is '{dr}'.")

            print(f"If cm is selected then make use value of k is less than 64.")
            print(f"The entered values are Feature: {f}, filter: {task_type_map[task_number]}, k: {k}, dimensionality "
                  f"reduction: {dr}, classifier: {classifier}")
            task_1_2_3(task_number=1, input_folder_path='Dataset', feature_model=f, k=k, test_folder_path=test_path,
                       classifier=classifier,
                       dim_red_technique=dr, output_folder='output', latent_semantics_file_name=None,
                       use_cached_input_images=True)

        if 3 < task_number < 7:
            task8(task_number)

        print("\nEnter the task number, to exit type e: ")
        task_number = input()
