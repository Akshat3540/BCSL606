import numpy as np, matplotlib.pyplot as plt

def lwr(x, X, y, t):
    W = np.diag(np.exp(-((X - x)**2).sum(1) / (2 * t**2)))
    return x @ np.linalg.solve(X.T @ W @ X, X.T @ W @ y)

np.random.seed(42)
X = np.linspace(0, 2*np.pi, 100)
y = np.sin(X) + 0.1*np.random.randn(100)

Xb = np.c_[X**0, X]
xt = np.linspace(0, 2*np.pi, 200)

yp = [lwr(i, Xb, y, 0.5) for i in np.c_[xt**0, xt]]

plt.scatter(X, y, c='red')
plt.plot(xt, yp)
plt.title('Locally Weighted Regression')
plt.grid()
plt.show()