import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame
num_f = df.select_dtypes(include=[np.number]).columns

for i, f in enumerate(num_f, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[f])
plt.show()

for i, f in enumerate(num_f, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[f], color='orange')
plt.show()