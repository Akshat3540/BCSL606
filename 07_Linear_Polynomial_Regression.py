import pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def reg(X, y, model, name):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    y_p = model.fit(X_tr, y_tr).predict(X_te)
    [plt.scatter(X_te, c, c=col, label=lbl) for c, col, lbl in [(y_te, "b", "Actual"), (y_p, "r", "Predicted")]]
    plt.title(name); plt.legend(); plt.show()
    print(f"{name}\nMSE: {mean_squared_error(y_te, y_p):.2f}\nR2: {model.score(X_te, y_te):.4f}\n")

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
reg(X[["AveRooms"]], y, LinearRegression(), "Linear Regression - California")

cols = ["mpg", "cyl", "disp", "hp", "wt", "acc", "year", "orig"]
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", sep=r'\s+', names=cols, na_values="?").dropna()
reg(df[["disp"]], df["mpg"], make_pipeline(PolynomialFeatures(2), LinearRegression()), "Polynomial Regression - Auto MPG")
