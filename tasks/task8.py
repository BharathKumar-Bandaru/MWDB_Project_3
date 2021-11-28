import numpy as np
from tasks.vafiles import *
from tasks.custom_svm import *
import icecream
from tasks.svm import *
from tasks.task4 import *


def task8(task_number):
    output_folder_7 = "svm_feedback_output_for_task_"
    output_image_features = []
    perform_task_6_7_flag = False
    if task_number == 4:
        print(f"Enter the database folder path:")
        folder_path = input()
        print(f"Enter the feature mode [cm, elbp, hog]:")
        feature = input()
        print(f"Enter the number of layers:")
        layers = int(input())
        print(f"Enter the number of hashes per layer:")
        hashes_per_layer = int(input())
        print(f"Enter the query_image: ")
        q_image_name = input()
        q_image_name = os.path.join(folder_path, q_image_name)
        print(f"Enter the number of similar images needed to be retrieved:")
        t = int(input())
        results = task4(input_folder=folder_path, feature_model=feature, num_layers=layers, num_hashes_per_layer=hashes_per_layer,
      query_image_path=q_image_name, num_similar_images_to_retrieve=t)
        print("\nDo you want to perfrom task 6 and 7? [y or n]:")
        key = input()
        perform_task_6_7_flag = True if key == "y" else False

        output_image_features = []
        for each in results:
            output_image_features.append((each.image_arr, each.filename, each.features))

    elif task_number == 5:
        # b - number of bits
        # Vectors that generate features.
        # Folder of images.
        # Feature model
        #  - input image
        # t - top t images
        print(f"Enter the database folder path:")
        folder_path = input()
        print("Enter the feature model [cm, elbp, hog]: ")
        feature = input()
        print("Enter the test image path:")
        test_image_path = input()
        print("Enter the t value(top t images):")
        t = int(input())
        print("Enter the number of bits(b):")
        b = int(input())

        test_image = get_images_and_attributes_from_folder(test_image_path)
        q_image_name = test_image[0]['filename']

        output_image_features = perform_va_files(folder_path, feature, test_image_path, t, b)
        print("\nDo you want to perfrom task 6 and 7? [y or n]:")
        key = input()
        perform_task_6_7_flag = True if key == "y" else False

    if perform_task_6_7_flag:
        print("Enter task 6 or 7:")
        task = int(input())

    if perform_task_6_7_flag and task == 6:
        return "task 6"

    elif perform_task_6_7_flag and task == 7:
        X_data, new_labels = mark_relevant_non_relevant(output_image_features)

        svm = SVM_custom(C=1, tol=0.01, max_iter=100, random_state=0, verbose=1)
        svm.train(X_data, new_labels)

        # icecream.ic([out[1] for out in output_image_features])
        # image_features = np.array([out[2] for out in output_image_features[10:]])
        # image_names = np.array([out[1] for out in output_image_features[10:]])
        images_with_attributes = get_images_and_attributes_from_folder(folder_path)
        images = [image_dict['image'] for image_dict in images_with_attributes]
        image_features = get_flattened_features_for_images(images, feature)
        image_names = [image_dict['filename'] for image_dict in images_with_attributes]


        labels = svm.predict(image_features)
        # icecream.ic(labels)
        weights = svm.get_weights(image_features)
        # icecream.ic(weights)
        max_ = np.argmax(weights, axis=1)
        # icecream.ic(max_)
        weights_ = []
        weights_non_ = []
        for i, val in enumerate(max_):
            if val == 1:
                weights_.append([weights[i][1], i])
            else:
                weights_non_.append([weights[i][0], i])
        weights_ = sorted(weights_, key = lambda t: t[0], reverse=True)
        weights_non_ = sorted(weights_non_, key = lambda t: t[0], reverse=True)
        # icecream.ic(weights_)
        img_result = []
        print("-----------")
        print(f"The top {t} images for input image {q_image_name}.")
        count = 1
        for i in weights_[:t]:
            img_result.append((images[i[1]], f'{count}_{image_names[i[1]]}'))
            print(image_names[i[1]])
            count+=1
        if(len(img_result) < t):
            count = 0
            while(len(img_result) <= t):
                img_result.append((images[weights_non_[count][1]], f'{count}_{image_names[weights_non_[count][1]]}'))
                print(image_names[weights_non_[count][1]])
                count+=1

        output_folder_7 = f"{output_folder_7}_{task_number}"
        save_images_by_clearing_folder(img_result, output_folder_7)

def mark_relevant_non_relevant(output_images):
    labels = []
    X_data = []
    names = []
    for each in output_images:
        print(f"Enter 1 for relevant and enter 0 for non-relevant for image {each[1]}:")
        val = int(input())
        labels.append(val)
        X_data.append(each[2])
        names.append(each[1])
    print(f"------------")
    for each_ in zip(names, labels):
        print(f"{each_[0]}: {each_[1]}")
    return np.array(X_data), np.array(labels)