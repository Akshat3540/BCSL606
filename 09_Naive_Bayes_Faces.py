import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = fetch_olivetti_faces(shuffle=True, random_state=42)

X_tr, X_te, y_tr, y_te = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

gnb = GaussianNB().fit(X_tr, y_tr)
y_p = gnb.predict(X_te)

print("Accuracy:", accuracy_score(y_te, y_p) * 100)
print(classification_report(y_te, y_p, zero_division=1))
print(confusion_matrix(y_te, y_p))

print("CV Accuracy:", cross_val_score(gnb, data.data, data.target, cv=5).mean() * 100)

fig, axes = plt.subplots(3, 5)
for ax, img, lbl, prd in zip(axes.ravel(), X_te, y_te, y_p):
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f"T:{lbl} P:{prd}")
    ax.axis('off')

plt.show()