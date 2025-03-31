import numpy as np
from scipy.linalg import eigh

A = np.array([
    [0, 2, 0, 0, 0, 0],
    [2, 0, 1, 0, 0, 0],
    [0, 1, 0, 6, 0, 0],
    [0, 0, 6, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0]
])

D = np.diag(A.sum(axis=1))   # 计算度

L = D - A   # 计算 Lap


print("度:\n", D)
print("lap:\n", L)


eigvals, eigvecs = eigh(L)



L_Lambda = 3  # 选前 3
phi_Lambda = eigvecs[:, :L_Lambda]

print("基函数 φ_Λ:\n", phi_Lambda)   # 基函





p_Lambda = np.random.rand(L_Lambda)   # 随机权重
p_Lambda /= np.linalg.norm(p_Lambda)  # 归

print("权重向量 p_Λ:\n", p_Lambda)


M_Lambda = 2
B_Lambda = np.random.rand(M_Lambda, L_Lambda)

psi_Lambda = B_Lambda @ phi_Lambda.T

print("Framelet 小波 ψ_(Λ,m):\n", psi_Lambda)


#fi(1)=[100000] fi(2)=[0100000]
#这两点往上的fi11=【1/根号2，1/根号2，0000】，。。。
#第二层开始有pesi方差，pesi=1/根号2（fi1^2-fi2^2)
#disanceng， 如果有十个：则pesi有C（2，10）个
#看截图！！！！！



