import scipy 
import numpy as np
import pickle
import networkx as nx
import random
import matplotlib.pyplot as plt 
from itertools import chain,combinations

def generate_graphlets(k):

    #calculates all 2^10 combinations of edges for K5 (or any other K)
    def generate_labelled_graphlets(G):
        s = list(G.edges())
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    G = nx.complete_graph(k)
    graphlets_labelled = generate_labelled_graphlets(G)
    graphlets=[]
    
    #these 1024 possibilities are narrowed down to 34 by checking for isomorphism
    for graph in graphlets_labelled:
        add=True
        G1 = nx.Graph()
        G1.add_nodes_from(G)
        G1.add_edges_from(list(list(graph)))
        for graphlet in graphlets:
            if nx.is_isomorphic(G1,graphlet):
                add=False
                break
        if add:
            graphlets.append(G1)
        
    return graphlets

def graphlet(graphs,k):
    
    graphlets=generate_graphlets(k)
    # initialization of the 34 dimensional feature vector
    features=[]

    for graph in graphs:
        
        # initialization of the 34 dimensional feature vector
        feature_vector=np.zeros(34)
        for i in range(1000):
            #randomly sampling nodes
            random_nodes = random.choices(np.array(graph.nodes()), k=5)
            subgraph = graph.subgraph(random_nodes)
            
            #checking for isomorphism
            for j,G in enumerate(graphlets):
                if nx.is_isomorphic(subgraph,G):
                    feature_vector[j]+=1
                    break
        features.append(feature_vector)
        
        
    
    features=np.array(features)


    gram_matrix=features.dot(features.T)
    return gram_matrix




