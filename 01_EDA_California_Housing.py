import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing(as_frame=True).frame
num_f = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(15, 10))
for i, f in enumerate(num_f, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[f], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {f}')
plt.tight_layout(); plt.show()

plt.figure(figsize=(15, 10))
for i, f in enumerate(num_f, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[f], color='orange')
    plt.title(f'Box Plot of {f}')
plt.tight_layout(); plt.show()

print("Outliers Detection:")
for f in num_f:
    q1, q3 = df[f].quantile([.25, .75])
    iqr = q3 - q1
    out = df[(df[f] < q1 - 1.5*iqr) | (df[f] > q3 + 1.5*iqr)]
    print(f"{f}: {len(out)} outliers")

print("\nDataset Summary:\n", df.describe())
