import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Datasets/autompg.csv")
X = df[['displacement']].values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sort_idx = np.argsort(X_test[:, 0])
X_test = X_test[sort_idx]
y_test = y_test[sort_idx]

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression")
print(f"MSE = {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"R2  = {r2_score(y_test, y_pred_lr):.4f}\n")

degrees = [2, 3, 4]
y_pred_poly = {}

for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)
    y_pred_poly[deg] = poly_reg.predict(X_test_poly)

    print(f"Polynomial Regression (degree={deg})")
    print(f"MSE = {mean_squared_error(y_test, y_pred_poly[deg]):.4f}")
    print(f"R2  = {r2_score(y_test, y_pred_poly[deg]):.4f}\n")

colors = {2: 'green', 3: 'purple', 4: 'orange'}
plt.figure(figsize=(10, 6))

plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Test Data')
plt.plot(X_test, y_pred_lr, color='red', linewidth=2, label='Linear Regression')

for deg in degrees:
    plt.plot(X_test, y_pred_poly[deg], color=colors[deg], linewidth=2, label=f'Degree {deg}')

plt.xlabel("Engine Displacement")
plt.ylabel("Miles per Gallon")
plt.title("Polynomial vs Linear Regression")
plt.legend()
plt.grid(True, alpha=0.6)
plt.show()
