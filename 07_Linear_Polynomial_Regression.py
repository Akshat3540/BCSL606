import pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def reg(X, y, model, name):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_tr, y_tr); y_p = model.predict(X_te)
    plt.scatter(X_te, y_te, c="blue", label="Actual")
    plt.scatter(X_te, y_p, c="red", label="Predicted")
    if "Linear" in name: plt.plot(X_te, y_p, c="red")
    plt.title(name); plt.legend(); plt.show()
    print(f"{name}\nMSE: {mean_squared_error(y_te, y_p)}\nR2: {r2_score(y_te, y_p)}\n")

c_df = fetch_california_housing(as_frame=True)
reg(c_df.data[["AveRooms"]], c_df.target, LinearRegression(), "Linear Regression - California")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
df = pd.read_csv(url, sep='\s+', names=["mpg", "cyl", "disp", "hp", "wt", "acc", "year", "orig"], na_values="?").dropna()
reg(df[["disp"]], df["mpg"], make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression()), "Polynomial Regression - Auto MPG")
