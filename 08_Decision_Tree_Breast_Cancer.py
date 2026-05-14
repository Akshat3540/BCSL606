import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
y_p = clf.predict(X_te)
print(f"Model Accuracy: {accuracy_score(y_te, y_p)*100:.2f}%")
prd = "Benign" if clf.predict([X_te[0]])[0] == 1 else "Malignant"
print(f"Predicted Class: {prd}")
plt.figure(figsize=(12,8)); plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names); plt.show()
