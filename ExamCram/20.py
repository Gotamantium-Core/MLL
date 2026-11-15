# Question 20 - Bias-Variance tradeoff via polynomial degrees (Housing)
# Polynomial regression of degrees 1..5: training vs validation MSE per degree, plotted.
# CRAM POINTS:
#   * Low degree -> high bias (both errors high). High degree -> high variance (train error low, val error explodes).
#   * Record training error AND validation error for every degree inside one loop.
#   * The sweet spot degree = where validation error is minimum.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("Datasets/housing.csv")

X = df.drop(columns = ["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

degrees = [1,2,3,4,5]
training_error = []
validation_error = []
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_p = poly.fit_transform(X_train_sc)
    X_test_p = poly.transform(X_test_sc)

    model = LinearRegression()
    model.fit(X_train_p,y_train)
    train_p = model.predict(X_train_p)
    y_p = model.predict(X_test_p)

    training_error.append(mean_squared_error(y_train, train_p))
    validation_error.append(mean_squared_error(y_test, y_p))
    print(f"\nDegree = {d}")
    print(f"Training error = {training_error[-1]: .4f}")
    print(f"Validation error = {validation_error[-1]: .4f}")

plt.figure(figsize=(7,5))
plt.plot(degrees, training_error, marker='o', label='Training Error', color="black")
plt.plot(degrees, validation_error, marker='s', label='Validation Error', color="red")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.grid(True)
plt.show()
