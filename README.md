import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 初始邻接矩阵
adjacency = np.array([
    [0, 2, 0, 0, 0, 0],
    [2, 0, 1, 0, 0, 0],
    [0, 1, 0, 6, 0, 0],
    [0, 0, 6, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0]
])

# 手动
levels = [
    [0, 0, 1, 1, 2, 2],  # 第一层聚类
    [0, 0, 1],  # 第二层聚类
    [0]  # 第三层聚类（最终合并）
]


# 计算新邻接矩阵的
def merge_adjacency(adjacency, labels):
    unique_labels = np.unique(labels)
    new_size = len(unique_labels)
    new_adjacency = np.zeros((new_size, new_size))

    label_map = {old: new for new, old in enumerate(unique_labels)}
    relabeled = np.vectorize(lambda x: label_map[x])(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            new_adjacency[relabeled[i], relabeled[j]] += adjacency[i, j]

    return new_adjacency


# 画图
def draw_graph(adjacency, labels, level):
    G = nx.Graph()
    num_nodes = adjacency.shape[0]

    # 加节点
    for i in range(num_nodes):
        G.add_node(i, label=f"C{labels[i]}")

    # 加边
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency[i, j] > 0:
                G.add_edge(i, j, weight=adjacency[i, j])

    pos = nx.spring_layout(G)
    colors = [plt.cm.Set1(l % 9) for l in labels]  # 颜色归一

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, edgecolors="black")
    plt.title(f"Level {level} Clustering Graph")
    plt.show()


# 逐层合并
current_adjacency = adjacency.copy()
for i, labels in enumerate(levels):
    draw_graph(current_adjacency, labels, i + 1)
    current_adjacency = merge_adjacency(current_adjacency, labels)
    print(f"Level {i + 1} adjacency matrix:")
    print(current_adjacency)
