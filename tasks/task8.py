import numpy as np


# Task 8 entry point
def ascossAlgorithm(similarity_matrix, n, m):
    graph = np.zeros((len(similarity_matrix), len(similarity_matrix)))
    score_matrix = np.random.rand(len(similarity_matrix), len(similarity_matrix))


    """ Construct the graph (V,E). Nodes with no edges contains values as -1"""
    print("Constructing similarity graph: ")
    for node in range(0, len(similarity_matrix)):
        similar_nodes_dict = findTopNSimilarObjects(node, similarity_matrix, n)
        for j in similar_nodes_dict.keys():
            graph[node][j] = similar_nodes_dict[j]

    c = 0.56

    itr=0
    while True:
        prev_score_matrix = score_matrix.copy()
        for i in range(0, len(similarity_matrix)):
            for j in range(0, len(similarity_matrix)):
                if i == j :
                    score_matrix[i][j] = 1
                else :
                    score = findSimilarityScore(graph,i, c, j, score_matrix)
                    score_matrix[i][j] = score
        sum = 0.0
        for i in range(0, len(score_matrix)):
            for j in range(0, len(score_matrix)):
                sum += np.abs(score_matrix[i][j] - prev_score_matrix[i][j])

        if sum == 0.0 and itr != 0:
            break
        itr += 1


    final_scores_dict = {}
    for i in range(0, len(score_matrix)):
        sum=0
        for j in range(0, len(score_matrix[i])):
            sum+= graph[i][j]
        final_scores_dict[i] = sum
    print(final_scores_dict)
    final_scores_dict = dict(sorted(final_scores_dict.items(), key=lambda item: item[1], reverse=True))
    count=0
    print("The most significant m objects are: ")
    for item in final_scores_dict.keys():
        print("Node: ", item)
        print("Value: ", final_scores_dict[item])
        count+=1
        if(count == m):
            break


def findSimilarityScore(graph, i, c, j, score_matrix):
    total_weight = 0
    for l in range(0, len(graph[i])):
        if (graph[i][l] != 0 and graph[l][i] != 0):
            total_weight += graph[i][l]
    edges_score = 0

    for k in range(0,len(graph[i])):
        if(graph[i][k] != 0 and graph[k][i] != 0):
            try:
                edges_score += (graph[i][k] / total_weight) * (1 - np.exp(-1 * graph[i][k])) * score_matrix[k][j]
            except OverflowError:
                edges_score = float('inf')

    return edges_score*c

def checkIncomingNode(graph, k):
    for i in range(0, len(graph)):
        if(graph[k][i] != 0):
            return True
    return False;

def findTopNSimilarObjects(node, similarity_matrix, n):
    ref_vector = similarity_matrix[node];
    x = {}
    index=0
    for value in ref_vector:
        x[index] = value
        index += 1
    x = dict(sorted(x.items(), key=lambda item: item[1]))

    result_dict = {}
    count = 0
    for j in x.keys():
        result_dict[j] = x[j]
        count += 1
        if (count == n):
            break
    return result_dict