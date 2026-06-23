import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = np.random.rand(100)
train, test = data[:50], data[50:]
labels = np.where(train <= 0.5, "Class1", "Class2")

def knn(td, tl, tp, k):
    nearest_labels = tl[np.argsort(np.abs(td - tp))[:k]]
    return Counter(nearest_labels).most_common(1)[0][0]

for k in [1, 2, 3, 4, 5, 20, 30]:
    print(f"\nk = {k}")
    
    preds = [knn(train, labels, p, k) for p in test]

    for i, (val, p) in enumerate(zip(test, preds), 51):
        print(f"Point x{i} (value {val:.4f}) was classified as {p}")

    plt.figure()
    plt.scatter(train, train*0)
    plt.scatter(test, test*0+1)
    plt.show()