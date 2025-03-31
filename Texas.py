import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

num_nodes = 183
num_edges = 325
np.random.seed(42)
edges = np.random.randint(0, num_nodes, size=(num_edges, 2))


G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)


adj_matrix = nx.adjacency_matrix(G)

adj_dense = adj_matrix.toarray()
print("邻接矩阵（稀疏格式）:\n", adj_matrix)


plt.figure(figsize=(8, 6))#显
nx.draw(G, with_labels=True, node_size=50, node_color="black", edge_color="gray")

plt.show()

