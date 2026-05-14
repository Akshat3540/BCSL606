import numpy as np, matplotlib.pyplot as plt
from collections import Counter
data = np.random.rand(100)
train, test = data[:50], data[50:]
labels = ["Class1" if x <= 0.5 else "Class2" for x in train]

def knn(tr, tl, pt, k):
    dists = sorted([(abs(pt-tr[i]), tl[i]) for i in range(50)])[:k]
    return Counter([l for d, l in dists]).most_common(1)[0][0]

k_vals = [1, 2, 3, 4, 5, 20, 30]
print("--- k-Nearest Neighbors Classification ---")
results = {}
for k in k_vals:
    print(f"Results for k = {k}:")
    preds = [knn(train, labels, p, k) for p in test]
    results[k] = preds
    for i, p in enumerate(preds, 51):
        print(f"Point x{i} (value: {test[i-51]:.4f}) is classified as {p}")
    print()

for k in k_vals:
    plt.figure(figsize=(10, 6))
    plt.scatter(train, [0]*50, c=['blue' if l=='Class1' else 'red' for l in labels], label="Train")
    for c, l in [('blue', 'Class1'), ('red', 'Class2')]:
        pts = [test[i] for i in range(50) if results[k][i] == l]
        plt.scatter(pts, [1]*len(pts), c=c, label=f"{l} (Test)", marker="x")
    plt.title(f"k-NN Classification (k={k})"); plt.legend(); plt.grid(True); plt.show()
