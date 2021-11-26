import os

from tasks.task1_2_3 import task_1_2_3
task_1_2_3(task_number = 2, input_folder_path = 'Dataset_100', feature_model = 'elbp', k =27, test_folder_path = 'Test_Dataset', classifier =  'ppr',
               dim_red_technique = 'pca', output_folder = 'output', latent_semantics_file_name = None,
               use_cached_input_images = True)

#task_1_2_3(task_number = 1, input_folder_path = 'Dataset_100', feature_model = 'elbp', k =19, test_folder_path = 'Test_Dataset', classifier =  'ppr',
               #dim_red_technique = 'pca', output_folder = 'output', latent_semantics_file_name = None,
               #use_cached_input_images = True)
"""
if __name__ == "__main__":
    dataset_path = "Dataset"
    print("Enter the task number, to exit type e: ")
    task_number = input()


    while task_number.isnumeric():
        task_number = int(task_number)

        if task_number == 1:

            print("Enter the dataset path:")
            data_path = input()
            print("\nFeatures: [cm, elbp, hog]")
            print("Enter the feature from the above list:")
            f = input()
            print(f"Enter the value of k:")
            k = int(input())
            while f == "cm" and k > 63:
                print(f"For entered value of feature:{f} the value of k:{k} should be less than 64.")
                print(f"Re-enter the value of k:")
                k = int(input())

            print("Dimensionality reduction: [pca, svd, lda, kmeans]")
            print("Enter one of the dimensionality reduction:")
            dr = input()
            if task_number == 0:
                task1_2(feature_model=f, filter='none', image_type=None, k=k, dim_red_technique=dr, folder_path=dataset_path, output_folder="output", latent_semantics_file_name=f"{f}_{dr}_latent_semantics.csv")
            else:
                print("Enter the type(x):")
                x = input()
                print(f"If cm is selected then make use value of k is less than 64.")
                print(f"The entered values are Feature: {f}, X: {x}, k: {k}, dimensionality reduction: {dr}")
                task1_2(feature_model=f, filter="type", image_type=x, k=k,
                    dim_red_technique=dr,
                    folder_path=dataset_path, output_folder='output',
                    latent_semantics_file_name=f"task1_{dr}_latent_semantics.csv")

        elif task_number == 2:
            print("Enter the dataset path:")
            data_path = input()
            print("\nFeatures: [cm, elbp, hog]")
            print("Enter the feature from the above list:")
            f = input()
            print("Enter the subject ID (y) [1-40]:")
            y = input()
            print(f"If cm is selected then make use value of k is less than 64.")
            print(f"Enter the value of k:")
            k = int(input())
            while f == "cm" and k > 63:
                print(f"For entered value of feature:{f} the value of k:{k} should be less than 64.")
                print(f"Re-enter the value of k:")
                k = int(input())

            print("Dimensionality reduction: [pca, svd, lda, kmeans]")
            print("Enter one of the dimensionality reduction:")
            dr = input()
            print(f"The entered values are Feature: {f}, Y: {y}, k: {k}, dimensionality reduction: {dr}")

            task1_2(feature_model=f, filter="subject_id", image_type=y, k=k,
                    dim_red_technique=dr,
                    folder_path=dataset_path, output_folder='output',
                    latent_semantics_file_name=f"task2_{dr}_latent_semantics.csv")


        print("\nEnter the task number, to exit type e: ")
        task_number = input()
"""