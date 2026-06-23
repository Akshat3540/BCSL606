import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
reduced = PCA(n_components=2).fit_transform(iris.data)

for i, target_name in enumerate(iris.target_names):
    plt.scatter(*reduced[iris.target == i].T, label=target_name)

plt.title('PCA on Iris Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()