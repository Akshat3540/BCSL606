import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
iris = load_iris()
pca_df = pd.DataFrame(PCA(2).fit_transform(iris.data), columns=['PC1', 'PC2'])
pca_df['L'] = iris.target

plt.figure(figsize=(8, 6))
for i, name in enumerate(iris.target_names):
    plt.scatter(pca_df[pca_df['L']==i]['PC1'], pca_df[pca_df['L']==i]['PC2'], label=name, color=['r','g','b'][i])
plt.title('PCA on Iris Dataset'); plt.xlabel('PC1'); plt.ylabel('PC2')
plt.legend(); plt.grid(); plt.show()
