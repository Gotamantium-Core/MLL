# Question 03 - Ridge & Lasso vs Linear Regression (Diabetes)
# LinearRegression vs RidgeCV vs LassoCV, tuned with cross-validation over alphas.
# CRAM POINTS:
#   * Ridge = L2 penalty, Lasso = L1 penalty (Lasso can zero-out coefficients).
#   * RidgeCV/LassoCV pick the best alpha_ by cross-validation - print ridge.alpha_.
#   * Compare MSE + R2 of all three with bar charts.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Datasets/diabetes.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)

y_pred_linear = linear.predict(X_test)

# Ridge Regression
alphas = [0.01, 0.1, 1, 10, 100]

ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)

# Lasso Regression
lasso = LassoCV(alphas=alphas, cv=4, random_state=42)
lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)

# Print errors for each
print("Errors for Linear Regression")
print(f"MSE = {mean_squared_error(y_test, y_pred_linear):.4f} \nR2 = {r2_score(y_test, y_pred_linear):.4f}\n")

print("Errors for Ridge Regression")
print(f"Best alpha = {ridge.alpha_}")
print(f"MSE = {mean_squared_error(y_test, y_pred_ridge):.4f} \nR2 = {r2_score(y_test, y_pred_ridge):.4f}\n")

print("Errors for Lasso Regression")
print(f"Best alpha = {lasso.alpha_}")
print(f"MSE = {mean_squared_error(y_test, y_pred_lasso):.4f} \nR2 = {r2_score(y_test, y_pred_lasso):.4f}\n")

models = ["Linear","Ridge","Lasso"]
mse = [mean_squared_error(y_test, y_pred_linear),mean_squared_error(y_test, y_pred_ridge),mean_squared_error(y_test, y_pred_lasso)]
r2 = [r2_score(y_test, y_pred_linear),r2_score(y_test, y_pred_ridge),r2_score(y_test, y_pred_lasso)]

plt.subplot(1,2,1)
plt.bar(models,mse,color="black")
plt.title("Mean Square Error")

plt.subplot(1,2,2)
plt.bar(models,r2,color='red')
plt.title("R2 Score")

plt.show()
