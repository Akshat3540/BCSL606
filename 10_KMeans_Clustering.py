import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

data = load_breast_cancer()

X = StandardScaler().fit_transform(data.data)
km = KMeans(n_clusters=2, random_state=42).fit(X)

print(confusion_matrix(data.target, km.labels_))
print(classification_report(data.target, km.labels_))

pca = PCA(2).fit_transform(X)
df = pd.DataFrame(pca, columns=['PC1', 'PC2'])

df['C'] = km.labels_
df['L'] = data.target

for h in ['C', 'L']:
    plt.figure()
    sns.scatterplot(data=df, x='PC1', y='PC2', hue=h)
    plt.show()