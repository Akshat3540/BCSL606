import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = np.random.rand(100)
train, test = data[:50], data[50:]
labels = ["Class1" if x <= 0.5 else "Class2" for x in train]

def knn(td, tl, tp, k):
    d = sorted([(abs(tp - td[i]), tl[i]) for i in range(len(td))])[:k]
    return Counter([l for _, l in d]).most_common(1)[0][0]

for k in [1,2,3,4,5,20,30]:
    print("\nk =", k)

    preds = [knn(train, labels, p, k) for p in test]

    for i, p in enumerate(preds):
        print(f"Point x{i+51} (value {test[i]:.4f}) was classified as {p}")

    plt.figure()
    plt.scatter(train, [0]*len(train))
    plt.scatter(test, [1]*len(test))

    plt.show()