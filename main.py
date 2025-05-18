from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

california = fetch_california_housing()

df = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target


X = df[["MedInc"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)


y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)


sort_idx = np.argsort(X_test.flatten())
X_sorted = X_test[sort_idx]
y_sorted = y_test[sort_idx]
y_pred_sorted = y_pred[sort_idx]



plt.scatter(X_test, y_test, color="lightblue", label="Actual data")
plt.plot(X_sorted, y_pred_sorted, color="red", label="Polynomial prediction")
plt.xlabel("Median Income")
plt.ylabel("House Price (in $100,000s)")
plt.title("Polynomial Regression on California Housing")
plt.legend()


plt.show()