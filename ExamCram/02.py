# Question 02 - Polynomial vs Linear Regression (Auto MPG)
# Predicts MPG from displacement: plain LinearRegression + PolynomialFeatures degrees 2,3,4.
# CRAM POINTS:
#   * poly.fit_transform on train, poly.transform on test (never fit on test!).
#   * Polynomial regression is still LinearRegression on the expanded features.
#   * Compare MSE + R2 for linear vs each degree, plot all curves together.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Datasets/autompg.csv")

# 'class' column holds MPG in this dataset
df = df[['displacement', 'class']]

X = df[['displacement']]
y = df[['class']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Linear regression
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)

y_prediction_linear = linearModel.predict(X_test)

print("Errors for Linear Regression")
print(f"MSE = {mean_squared_error(y_test, y_prediction_linear): .4f} \nR2 = {r2_score(y_test, y_prediction_linear): .4f}\n")

# Polynomial Regression
degrees = [2, 3, 4]
models = {}
colors = ["blue","violet","pink"]
for deg in degrees:
    poly = PolynomialFeatures(degree=deg)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Errors for Polynomial Regression (degree={deg})")
    print(f"MSE = {mse: .4f} \nR2 = {r2: .4f}\n")

    models[deg] = (poly, model)

size = min(200,len(X_test))
plt.scatter(X_test[:size],y_test[:size],color='red',label='Data points')
xline = np.linspace(float(X_test.min().iloc[0]),float(X_test.max().iloc[0]),100).reshape(-1,1)
y_lr = linearModel.predict(xline)
plt.plot(xline,y_lr,color="black",label="Std Linear Regression")

for i in range(3):
    xline_p = models[degrees[i]][0].transform(xline)
    plt.plot(xline,models[degrees[i]][1].predict(xline_p),color=colors[i],label = f"Degree {degrees[i]}")

plt.legend()
plt.grid()
plt.show()
