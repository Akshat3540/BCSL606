import numpy as np, matplotlib.pyplot as plt

def lwr(x, X, y, t):
    x = np.array(x)
    W = np.diag(np.exp(-np.sum((X-x)**2, axis=1)/(2*t**2)))
    θ = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    return x @ θ

np.random.seed(42)

X = np.linspace(0, 2*np.pi, 100)
y = np.sin(X) + 0.1*np.random.randn(100)

Xb = np.c_[np.ones(100), X]
xt = np.linspace(0, 2*np.pi, 200)
xtb = np.c_[np.ones(200), xt]

yp = [lwr(i, Xb, y, 0.5) for i in xtb]

plt.scatter(X, y, color='red')
plt.plot(xt, yp)
plt.title('Locally Weighted Regression')
plt.grid()
plt.show()