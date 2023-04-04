import gensim.models
import networkx as nx
import numpy as np
import random
from gensim.models import Word2Vec

#Random walk
def deepwalk_walk(G,walk_length, start_node):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk
#Generate a random walk sequence
def _simulate_walks(G,nodes, num_walks, walk_length):
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:
            walks.append(deepwalk_walk(G,walk_length = walk_length, start_node=v))
    return walks
def get_embeddings(edgelist_file,save_file_node,save_file_embedding):
    G=nx.read_edgelist(edgelist_file,create_using=nx.Graph())
    nodes = list(G.nodes())
    walks = _simulate_walks(G,nodes, num_walks=80, walk_length=10)
    model = gensim.models.Word2Vec(walks)
    pnodes_map = model.wv.index_to_key
    file = open(save_file_node, mode='w')
    for i in pnodes_map:
        file.write(str(i) +'\n')
    file.close()

    np.savetxt(save_file_node,pnodes_map,fmt='%d')
    ppi_embeddings = model.wv.vectors
    np.savetxt(save_file_embedding,ppi_embeddings,fmt='%.4f')
    return
edgelist_file = './data/ddi.txt'
save_file_node = './data/dnode.txt'
save_file_embedding = './data/ddii_embedding.txt'
get_embeddings(edgelist_file,save_file_node,save_file_embedding)
#***********************************
#'edgelist_file' in this code is the integrated drug interaction network
#'save_file_node' and 'save_file_embedding' are your own paths for storing the id of node(drug) and feature embedding obtained using the Deepwalk algorithm.
#In this paper, it is also necessary to use the Deepwalk algorithm for the integrated protein interaction network, so as to obtain protein feature embedding. 
#So you just change the value of 'edgelist_file' to the integrated drug interaction network, which is 'edgelist_file = './data/ddi.txt''.
#***********************************

