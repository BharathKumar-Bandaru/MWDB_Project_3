import math
from tasks.input_output import *
import numpy as np
from tasks.features import *


def compute_number_of_bits(b, d):
    # v = number of vectors
    # b = number of bits
    # d = number of dimensions
    bits = []

    for j in range(1, d+1):
        if j <= b%d:
            bj = math.floor(b/d) + 1
        else:
            bj = math.floor(b/d)+0
        bits.append(bj)

    return bits

def distribute_partition_points(feature_vector, dim_num, bits):
    #ex: for dim_num = 0 concatenate the first col values of all the data
    col = []
    for i in range(0, len(feature_vector)):
        col.append(feature_vector[i][dim_num])
    col.sort()

    partition_points = []
    partition_points.append(int(col[0]))
    # print("Col size is: ", len(col))
    # print("Bits[dim_num] is: ",bits[dim_num])
    num_points_per_region = len(col)//math.pow(2, bits[dim_num])
   # print("Number of points per region ", num_points_per_region)
    count = 0
    for i in range(1, len(col)-1):
        count+=1
        if count == num_points_per_region:
            partition_points.append(int((col[i] + col[i+1])/2.0))
            count = 0
    partition_points.append(int(col[len(col)-1] + 1.0))
    return partition_points


def bin(n, length):
    i = 1 << length
    ans = ""
    while (i > 0):
        if ((n & i) != 0):
            ans = ans + "1"
            #print("1", end="")
        else:
            ans = ans + "0"
            #print("0", end="")

        i = i // 2
    return ans

def compute_region(vector, partition_points):
    regions = []
    # print(len(vector))
    for j in range(0, len(vector)):
        flag = 0
        region = 0
        # print("Length of partition points is: ", len(partition_points[j]))
        for i in range(0, len(partition_points[j])-1):
            if(partition_points[j][i] <= int(vector[j]) and int(vector[j]) < partition_points[j][i+1]):
                regions.append(i)
                flag = 1
                break
        #Handle edge case
        if(flag == 0):
            if(int(vector[j]) < partition_points[j][0]):
                regions.append(partition_points[j][0])
            elif(int(vector[j]) >= partition_points[j][len(partition_points)-1]):
                regions.append(partition_points[j][len(partition_points)-1])


    return regions

def compute_bit_string(vector, partition_points, bits):
    final_rep = ""
    regions = []
    for j in range(0, len(vector)):
        flag = 0

        for i in range(0, len(partition_points[j])-1):
            if (partition_points[j][i] <= vector[j] and vector[j] < partition_points[j][i + 1]):
                regions.append(i)
                bit_string = bin(partition_points[j][i], bits[j])
                final_rep = final_rep + bit_string
                flag = 1
                break
        # Handle edge case
        if (flag == 0):
            if (vector[j] < partition_points[j][0]):
                regions.append(partition_points[j][0])
                bit_string = bin(partition_points[j][0], bits[j])
                final_rep = final_rep + bit_string
            elif (vector[j] >= partition_points[j][len(partition_points[j]) - 1]):
                regions.append(partition_points[j][len(partition_points[j]) - 1])
                bit_string = bin(partition_points[j][len(partition_points[j]) - 1], bits[j])
                final_rep = final_rep + bit_string

    return final_rep, regions


def compute_lower_bound(vector, vq, partition_points):
    li = 0
    rq = compute_region(vq, partition_points)
    ri = compute_region(vector, partition_points)

    for j in range(0, len(rq)):
        if ri[j] < rq[j]:
            lij = vq[j] - partition_points[j][ri[j]+1]
        elif ri[j] == rq[j]:
            lij = 0
        else:
            lij = partition_points[j][ri[j]] - vq[j]
        li += lij*lij

    return math.sqrt(li)

def compute_euclidean(v1, v2):
    dist = 0
    for i in range(0, len(v1)):
        dist += (v1[i]-v2[i])*(v1[i]-v2[i])
    return dist

def compute_manhattan(v1, v2):
    dist = 0
    for i in range(0, len(v1)):
        dist += abs(v1[i]-v2[i])
    return dist
def Candidate(d, i, dst, ans, t):
    if d < dst[t-1]:
        dst[t-1] = d
        ans[t-1] = i

    #SortOnDst(ans, dst, t);
    ans = [dis for _, dis in sorted(zip(dst, ans))]
    dst.sort()
    return ans, dst

def simple_search_algorithm(feature_vectors, vq, t, partition_points):
    ans = []
    dst = []
    buckets = 0
    for k in range(0, t):
        dst.append(float('inf'))
        ans.append(0)


    d = float('inf')
    idx = 0
    for i in range(0, len(feature_vectors)):
        li = compute_lower_bound(feature_vectors[i], vq, partition_points)

        if li < d or idx < t:
            buckets += 1
            d = compute_euclidean(vq, feature_vectors[i])
            idx+=1
            ans, dst = Candidate(d, i, dst, ans, t)

    return ans, dst, buckets

def calculate_distances(original, test):
    result = {}
    i = 0
    for img in original:
        result[i] = compute_manhattan(img, test)
        i+=1

    actual_result = sorted(result.items(), key=lambda x: x[1])

    return actual_result


def perform_va_files(folder_path, feature, test_image_path, t, b):
    image_data_attributes = get_images_and_attributes_from_folder(folder_path)
    images = [img['image'] for img in image_data_attributes]
    image_names = [img['filename'] for img in image_data_attributes]
    image_features = get_flattened_features_for_images(images, feature)

    d = len(image_features[0])
    bits = compute_number_of_bits(b, d)
    overall_partition_list = []
    for i in range(d):
        partition_points = distribute_partition_points(image_features, i, bits)
        overall_partition_list.append(partition_points)

    approximations = []
    print("Calculating approximations: ")
    for i in range(0, len(image_features)):
        bit_string, regions = compute_bit_string(image_features[i], overall_partition_list, bits)
        approximations.append(bit_string)
    approx_set = set(approximations)

    test_dataset = get_images_and_attributes_from_folder(test_image_path)
    actual = []
    test_type = test_dataset[0]['type']
    test_subject_id = test_dataset[0]['subject_id']
    test_image_id = test_dataset[0]['image_id']
    test_file_name = test_dataset[0]['filename']
    actual.append(test_type)
    actual.append(test_subject_id)
    query_vector = np.array(get_flattened_features_for_images([test_dataset[0]['image']], feature))
    ans, dst, buckets = simple_search_algorithm(image_features, query_vector[0], t, overall_partition_list)
    expected_files = []
    actual_result = calculate_distances(image_features, query_vector[0])

    idx = 0
    print("Expected outputs are: ")
    for x in actual_result:
        expected_files.append(image_names[x[0]])
        print(image_names[x[0]])
        idx += 1
        if idx == t:
            break
    print("Output vectors are saved to the Output_VAFiles folder: ")
    output = []
    output_features = []
    output.append((test_dataset[0]['image'], "input-" + test_file_name))
    false_positives = 0
    misses = 0
    count = 0
    for a in ans:
        if (image_data_attributes[a]['type'] not in actual and image_data_attributes[a]['subject_id'] not in actual):
            false_positives += 1
        if (image_data_attributes[a]['filename'] not in expected_files):
            misses += 1
        output.append((images[a], image_names[a]))
        output_features.append((images[a], image_names[a], image_features[a]))
        count+=1
    save_images_by_clearing_folder(output, "Output_VAFiles")

    print("Number of unique approximations considered are: ", len(approx_set))
    print("Number of overall approximations considered are ", len(approximations))
    print("Number of bytes required for index structure is: ", len(approximations) * len(approximations[0]) / 8)
    print("Number of buckets searched are: ", buckets)
    print("The number of false positives are: ", false_positives)
    print("The miss rate is: ", misses / len(expected_files))
    return output_features


# #Test function
# #read dataset from folder path. Output is a list of dictionaries
# image_data = get_images_and_attributes_from_folder("Dataset")
# images = []
# file_names = []
# for img_dict in image_data:
#     images.append(img_dict['image'])
#     file_names.append(img_dict['filename'])
# feature_vetors = []
# for im in images:
#     input_vector = computer_hog(im)
#     feature_vetors.append(input_vector)
#
#
# #take b value as input"
# b = 5000
#
# d = len(feature_vetors[0])
#
# bits = compute_number_of_bits(b, d)
#
# overall_partition_list = []
# #Get partitionpartition_points[len(partition_points)-1] points for each dimension
# #overall_partition_list contains patition points list for every dimension.
# print(len(bits))
# print(len(feature_vetors[0]))
#
# for i in range(0, d):
#     partition_points = distribute_partition_points(feature_vetors, i, bits)
#     #print(partition_points)
#     overall_partition_list.append(partition_points)
#
#
# #Get approximations for each vector/datapoint
# approximations = []
# print("Calculating approximations: ")
# for i in range(0, len(feature_vetors)):
#     bit_string, regions = compute_bit_string(feature_vetors[i], overall_partition_list, bits)
#     approximations.append(bit_string)
# approx_set = set(approximations)
#
#
# test_dataset = get_images_and_attributes_from_folder("TestVA")
# actual = []
# test_type = test_dataset[0]['type']
# test_subject_id = test_dataset[0]['subject_id']
# test_image_id = test_dataset[0]['image_id']
# test_file_name = test_dataset[0]['filename']
# actual.append(test_type)
# actual.append(test_subject_id)
#
# query_vector = computer_hog(test_dataset[0]['image'])
# t = 10
# ans, dst, buckets = simple_search_algorithm(feature_vetors, query_vector, t, overall_partition_list)
# expected_files = []
# actual_result = calculate_distances(feature_vetors, query_vector)
#
# idx = 0
# print("Expected outputs are: ")
# for x in actual_result:
#     expected_files.append(file_names[x[0]])
#     print(file_names[x[0]])
#     idx+=1
#     if idx == t:
#         break
#
#
# print("Output vectors are saved to the Output_VAFiles folder: ")
# output = []
# output.append((test_dataset[0]['image'], "input-"+test_file_name))
# false_positives = 0
# misses = 0
# for a in ans:
#     if(image_data[a]['type'] not in actual and image_data[a]['subject_id'] not in actual):
#         false_positives += 1
#     if(image_data[a]['filename'] not in expected_files) :
#         misses += 1
#     output.append((images[a], file_names[a]))
# save_images_by_clearing_folder(output, "Output_VAFiles")
#
# print("Number of unique approximations considered are: ", len(approx_set))
# print("Number of overall approximations considered are ", len(approximations))
# print("Number of bytes required for index structure is: ", len(approximations)*len(approximations[0])/8)
# print("Number of buckets searched are: ", buckets)
# print("The number of false positives are: ",false_positives)
# print("The miss rate is: ",misses/len(expected_files))