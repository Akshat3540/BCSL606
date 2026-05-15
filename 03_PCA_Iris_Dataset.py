import pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()

pca_df = pd.DataFrame(
    PCA(2).fit_transform(iris.data),
    columns=['PC1', 'PC2']
)

pca_df['L'] = iris.target

for i in range(3):
    plt.scatter(
        pca_df[pca_df['L'] == i]['PC1'],
        pca_df[pca_df['L'] == i]['PC2'],
        label=iris.target_names[i]
    )
plt.title('PCA on Iris Dataset'); plt.xlabel('PC1'); plt.ylabel('PC2')
plt.legend()
plt.show()