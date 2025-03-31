from sknetwork.clustering import Louvain
from sknetwork.visualization import visualize_graph
import scipy.sparse as sp
import numpy as np

adjacency = np.array([
    [0,2,0,0,0,0],
    [2,0,1,0,0,0],
    [0,1,0,6,0,0],
    [0,0,6,0,1,0],
    [0,0,0,1,0,1],
    [0,0,0,0,1,0],
])

adjacency_sparse = sp.csr_matrix(adjacency)

def louvain_recursive_clustering(adjacency, iteration=0, prev_num_communities=None):
    louvain = Louvain()
    labels = louvain.fit_predict(adjacency)  #聚类
    num_communities = len(set(labels))

    # 生成
    svg_image = visualize_graph(adjacency, labels=labels)
    filename = f"louvain_{time.strftime('%Y%m%d_%H%M%S')}_iter_{iteration}_com_{num_communities}.svg"

    with open(f"louvain_iteration_{iteration}.svg", "w", encoding="utf-8") as f:
        f.write(svg_image)
    print(f"{filename} 生成完成")

    print(f"Iteration {iteration}: {num_communities} communities detected.")

    if prev_num_communities is not None and num_communities >= prev_num_communities:
        print("社区数量不再减少，递归终止。")
        return labels

    # 构造新
    new_adjacency = np.zeros((num_communities, num_communities))

    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] != labels[j]:
                new_adjacency[labels[i], labels[j]] += adjacency[i, j]

    new_adjacency_sparse = sp.csr_matrix(new_adjacency)
    if new_adjacency_sparse.nnz == 0:
        print("没有新的社区连接，递归终止。")
        return labels

    # 递归
    return louvain_recursive_clustering(new_adjacency_sparse, iteration + 1, num_communities)

final_labels = louvain_recursive_clustering(adjacency_sparse)
print("最终社区划分:", final_labels)