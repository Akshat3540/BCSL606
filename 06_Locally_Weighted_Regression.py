import numpy as np, matplotlib.pyplot as plt
def lwr(x, X, y, tau):
    W = np.diag(np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2)))
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    return x @ theta

np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)
X_b = np.c_[np.ones(100), X]
x_t = np.linspace(0, 2 * np.pi, 200)
x_tb = np.c_[np.ones(200), x_t]
y_p = [lwr(xi, X_b, y, 0.5) for xi in x_tb]

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Train', alpha=0.7)
plt.plot(x_t, y_p, color='blue', label='LWR Fit', linewidth=2)
plt.title('Locally Weighted Regression'); plt.legend(); plt.grid(alpha=0.3); plt.show()
