import numpy as np
from tasks.vafiles import *
from tasks.custom_svm import *
from tasks.svm import *
from tasks.task4 import *
from tasks.decision_trees import *
from scipy.spatial.distance import cityblock, euclidean


def task8(task_number):
    task = None
    if task_number == 6 or task_number == 7:
        print(f"Enter the task number 4 or 5 as a pre-requisite to the task {task_number}")
        task = task_number
        task_number = int(input())
    output_folder_7 = "_feedback_output_for_task_"

    output_image_features = []
    perform_task_6_7_flag = False
    print(f"Enter the database folder path:")
    folder_path = input()
    print(f"Enter the feature mode [cm, elbp, hog]:")
    feature = input()
    print("Enter the test image path:")
    test_image_path = input()
    print("Enter the test image name:")
    test_image_name = input()

    if task_number == 4:
        print(f"Enter the number of layers:")
        layers = int(input())
        print(f"Enter the number of hashes per layer:")
        hashes_per_layer = int(input())
        print("Enter the test image path:")
        q_image_name = os.path.join(test_image_path, test_image_name)
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
        print("Enter the t value(top t images):")
        t = int(input())
        print("Enter the number of bits(b):")
        b = int(input())

        q_image_name = os.path.join(test_image_path, test_image_name)


        output_image_features = perform_va_files(folder_path, feature, q_image_name, t, b)
        if task is None:
            print("\nDo you want to perfrom task 6 and 7? [y or n]:")
            key = input()
            perform_task_6_7_flag = True if key == "y" else False
        if task is not None:
            perform_task_6_7_flag = True

    if perform_task_6_7_flag and task is None:
        print("Enter task 6 or 7:")
        task = int(input())
        query_image = get_image_arr_from_file(q_image_name)
        query_image_features = get_flattened_features_for_images([query_image], feature)
        query_image_features = query_image_features[0]

        image_features = np.array([each[2] for each in output_image_features])
        image_names = np.array([each[1] for each in output_image_features])
        images = np.array([each[0] for each in output_image_features])
        # icecream.ic(image_features.shape, image_names.shape, images.shape, image_names)

    if perform_task_6_7_flag and task == 6:
        relevant_images, non_relevant_images, relevant_images_id, non_relevant_image_id = mark_relevant_non_relevant(
            output_image_features)

        X_data = np.append(relevant_images, non_relevant_images, axis=0)
        label = np.array([1] * len(relevant_images_id) + [0] * len(non_relevant_image_id))

        idx = np.random.permutation(len(X_data))
        X_data, new_labels = X_data[idx], label[idx]

        print("Enter the value of K for dim_reduction:")
        k = int(input())

        l, right_factor_matrix = perform_dim_red("pca", X_data, k)
        X_data = np.array(np.matmul(X_data, np.transpose(right_factor_matrix)))
        image_features = np.array(np.matmul(image_features, np.transpose(right_factor_matrix)))

        dataset_features = add_label_to_image_arr(X_data, new_labels)

        dc = DecisionTreeClassifier()
        root = dc.make_tree(dataset_features)

        predicitons = []
        for i in range(len(image_features)):
            pre = dc.predict(image_features[i], root)
            predicitons.append(pre)
        predicitons = np.array(predicitons)

        relevant_test_images = np.where(predicitons == 1)
        non_relevant_test_images = np.where(predicitons == 0)
        query_image_features = np.array(np.matmul(query_image_features, np.transpose(right_factor_matrix)))

        ordered_relevant = get_top_k(query_image_features, image_features[relevant_test_images],
                                     image_names[relevant_test_images], images[relevant_test_images])
        ordered_irrelevant = get_top_k(query_image_features, image_features[non_relevant_test_images],
                                       image_names[non_relevant_test_images], images[non_relevant_test_images])

        final_predictions = ordered_relevant[:t]

        if len(final_predictions) < t - 1:
            final_predictions = np.append(final_predictions, ordered_irrelevant[: t - len(final_predictions)], axis=0)

        for i in final_predictions:
            print(i[1])
        output_folder_7 = f"decision_tree{output_folder_7}{task_number}"
        img_result = []
        for i, each in enumerate(final_predictions):
            img_result.append((each[2], f"{i + 1}_{each[1]}"))
        print("The similar images after decision tree classifier.")

        save_images_by_clearing_folder(img_result, output_folder_7)


    elif perform_task_6_7_flag and task == 7:
        relevant_images, non_relevant_images, relevant_images_id, non_relevant_image_id = mark_relevant_non_relevant(output_image_features)

        X_data = np.append(relevant_images, non_relevant_images, axis=0)
        label = np.array([1] * len(relevant_images_id) + [0] * len(non_relevant_image_id))

        idx = np.random.permutation(len(X_data))
        X_data, new_labels = X_data[idx], label[idx]

        svm = SVM_custom(C=1, tol=0.01, max_iter=100, random_state=0, verbose=1)
        svm.train(X_data, new_labels)

        predicitons = svm.predict(image_features)

        relevant_test_images = np.where(predicitons == 1)
        non_relevant_test_images = np.where(predicitons == 0)

        ordered_relevant = get_top_k(query_image_features, image_features[relevant_test_images], image_names[relevant_test_images], images[relevant_test_images])
        ordered_irrelevant = get_top_k(query_image_features, image_features[non_relevant_test_images], image_names[non_relevant_test_images], images[non_relevant_test_images])

        final_predictions = ordered_relevant[:t]

        if len(final_predictions) < t-1:
            final_predictions = np.append(final_predictions, ordered_irrelevant[: t - len(final_predictions)], axis=0)

        for i in final_predictions:
            print(i[1])

        output_folder_7 = f"svm{output_folder_7}{task_number}"
        img_result = []
        for i, each in enumerate(final_predictions):
            img_result.append((each[2], f"{i+1}_{each[1]}"))
        save_images_by_clearing_folder(img_result, output_folder_7)

        # weights = svm.get_weights(image_features)
        # max_ = np.argmax(weights, axis=1)
        # weights_ = []
        # weights_non_ = []
        # for i, val in enumerate(max_):
        #     if val == 1:
        #         weights_.append([weights[i][1], i])
        #     else:
        #         weights_non_.append([weights[i][0], i])
        # weights_ = sorted(weights_, key = lambda t: t[0], reverse=True)
        # weights_non_ = sorted(weights_non_, key = lambda t: t[0], reverse=True)
        #
        # img_result = []
        # print("-----------")
        # print(f"The top {t} images for input image {q_image_name}.")
        # count = 1
        # for i in weights_[:t]:
        #     img_result.append((images[i[1]], f'{count}_{image_names[i[1]]}'))
        #     print(image_names[i[1]])
        #     count+=1
        # if(len(img_result) < t):
        #     count = 0
        #     while(len(img_result) <= t):
        #         img_result.append((images[weights_non_[count][1]], f'{count}_{image_names[weights_non_[count][1]]}'))
        #         print(image_names[weights_non_[count][1]])
        #         count+=1


        # icecream.ic(final_predictions)

        # icecream.ic([out[1] for out in output_image_features])
        # image_features = np.array([out[2] for out in output_image_features[10:]])
        # image_names = np.array([out[1] for out in output_image_features[10:]])

        # images_with_attributes = get_images_and_attributes_from_folder(folder_path)
        # images = [image_dict['image'] for image_dict in images_with_attributes]
        # image_features = get_flattened_features_for_images(images, feature)
        # image_names = [image_dict['filename'] for image_dict in images_with_attributes]


        # labels = svm.predict(image_features)
        # # icecream.ic(labels)
        # weights = svm.get_weights(image_features)
        # # icecream.ic(weights)
        # max_ = np.argmax(weights, axis=1)
        # # icecream.ic(max_)
        # weights_ = []
        # weights_non_ = []
        # for i, val in enumerate(max_):
        #     if val == 1:
        #         weights_.append([weights[i][1], i])
        #     else:
        #         weights_non_.append([weights[i][0], i])
        # weights_ = sorted(weights_, key = lambda t: t[0], reverse=True)
        # weights_non_ = sorted(weights_non_, key = lambda t: t[0], reverse=True)
        # # icecream.ic(weights_)
        # img_result = []
        # print("-----------")
        # print(f"The top {t} images for input image {q_image_name}.")
        # count = 1
        # for i in weights_[:t]:
        #     img_result.append((images[i[1]], f'{count}_{image_names[i[1]]}'))
        #     print(image_names[i[1]])
        #     count+=1
        # if(len(img_result) < t):
        #     count = 0
        #     while(len(img_result) <= t):
        #         img_result.append((images[weights_non_[count][1]], f'{count}_{image_names[weights_non_[count][1]]}'))
        #         print(image_names[weights_non_[count][1]])
        #         count+=1
        #
        # output_folder_7 = f"{output_folder_7}_{task_number}"
        # save_images_by_clearing_folder(img_result, output_folder_7)


def mark_relevant_non_relevant(output_images):
    image_names = [f"{i+1}. {each[1]}" for i, each in enumerate(output_images)]
    for i in image_names:
        print(i)

    print("Enter the relevant image id's from the above list:")
    relevant_images_id = np.array(input().split()).astype(int) - 1

    print("Enter the non-relevant image id's from the above list:")
    non_relevant_image_id = np.array(input().split()).astype(int) - 1

    revelant_images = np.array([each[2] for each in np.array(output_images)[relevant_images_id]])
    non_revelant_images = np.array([each[2] for each in np.array(output_images)[non_relevant_image_id]])

    print("Relevant Images:")
    print(np.array([each[1] for each in np.array(output_images)[relevant_images_id]]))
    print("Irr-Relevant Images:")
    print(np.array([each[1] for each in np.array(output_images)[non_relevant_image_id]]))

    return revelant_images, non_revelant_images, relevant_images_id, non_relevant_image_id

    # labels = []
    # X_data = []
    # names = []
    #
    # for each in output_images:
    #     print(f"Enter 1 for relevant and enter 0 for non-relevant for image {each[1]}:")
    #     val = int(input())
    #     labels.append(val)
    #     X_data.append(each[2])
    #     names.append(each[1])
    # print(f"------------")
    # for each_ in zip(names, labels):
    #     print(f"{each_[0]}: {each_[1]}")
    # return np.array(X_data), np.array(labels)

def get_top_k(q_image, image_features, image_names, image):
    top_k_images = []
    for i in range(len(image_features)):
        top_k_images.append([euclidean(q_image, image_features[i]), image_names[i], image[i]])

    top_k_images = sorted(top_k_images, key = lambda k: k[0])
    return np.array(top_k_images)