import numpy as np
from numpy import genfromtxt
from numpy import linalg

# Task 9 entry point.
def task9(similarity_matrix_file_path, n, m, seed_nodes = [0], beta = 0.15):
    """
    This task calculates Robust Personalized Page Rank scores for each node (subject) and reports the top most significant m nodes (subjects)
    similarity_matrix_file_path: subject-subject similarity matrix file path
    n: number of similar nodes (subjects) to connect each node (subject) with
    m: number of most significant nodes (subjects) to be retrieved
    seed_nodes: list of subject IDs to be used as seed nodes (0-indexed)
    beta (optional): random jump probability
    """
    print('Running Personalized Page Rank algorithm..')

    seed_nodes = [s-1 for s in seed_nodes] #converting 1-indexed values to 0-indexed values

    similarity_matrix = genfromtxt(similarity_matrix_file_path, delimiter=',')
    
    numNodes = len(similarity_matrix)

    print('Constructing graph from the similarity matrix..')
    TG = np.zeros((numNodes, numNodes))
    
    for node in range(len(similarity_matrix)):
        similar_nodes = getTopSimilarObjects(node, n, similarity_matrix)
        for similar_node in similar_nodes:
            TG[node][similar_node] = 1/n

    print('Calculating Robust Personalized Page Rank scores relative to the seed nodes..')
    
    #Step 1
    Pi_dict = {} #key = seed_node, value = Pi_i vector

    for seed_node in seed_nodes:
        Si = np.matrix(np.zeros((numNodes, 1)))
        Si[seed_node] = 1

        a = ((1 - beta) * TG)
        I = np.identity(TG.shape[0])
        inv = linalg.inv(I-a)
        Pi_i = inv * beta * Si
        Pi_dict[seed_node] = Pi_i

    #Step 2
    seed_scores_dict = {}
    for seed_node in seed_nodes:
        Pi_i = Pi_dict[seed_node]
        summation = 0
        for node in seed_nodes:
            summation += Pi_i[node].item()
        seed_scores_dict[seed_node] = summation
        
    #Step 3
    S_crit = []
    max_seed_score = seed_scores_dict[max(seed_scores_dict, key = seed_scores_dict.get)]
    for seed_node in seed_nodes:
        if seed_scores_dict[seed_node] == max_seed_score:
            S_crit.append(seed_node)
        
    #Step 4
    #RPR-2 Scores (Robust Personalized Page Rank Scores)
    PPR_Scores_matrix = sum(Pi_dict.values()) / len(Pi_dict) #Taking average of all Pi values
    PPR_Scores = [scoreMatrix.item() for scoreMatrix in PPR_Scores_matrix]
    
    #Returning the result
    m_most_significant_nodes = np.argsort(PPR_Scores)[::-1][:m]

    #1-indexed nodes
    [(index + 1, PPR_Scores[index]) for index in m_most_significant_nodes]
    
    print('Top {0} significant subjects IDs with their scores:'.format(m))

    for node in m_most_significant_nodes:
        print(f'Subject {node+1} : {PPR_Scores[node]}')
    
    return m_most_significant_nodes

def getTopSimilarObjects(node, n, similarity_matrix):    
    nodes_in_order = np.argsort(similarity_matrix[node])
    top_similar_nodes = nodes_in_order[:n]
    return top_similar_nodes