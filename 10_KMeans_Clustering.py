import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

data = load_breast_cancer()
X_s = StandardScaler().fit_transform(data.data)
km = KMeans(n_clusters=2, random_state=42).fit(X_s)
print(f"Confusion Matrix:\n{confusion_matrix(data.target, km.labels_)}\n\nReport:\n{classification_report(data.target, km.labels_)}")

pca = PCA(2).fit_transform(X_s)
df = pd.DataFrame(pca, columns=['PC1', 'PC2'])
df['C'], df['L'] = km.labels_, data.target

for hue, title in [('C', 'K-Means Clustering'), ('L', 'True Labels')]:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue=hue, palette='Set1' if hue=='C' else 'coolwarm', s=100, edgecolor='black', alpha=0.7)
    plt.title(title); plt.show()
