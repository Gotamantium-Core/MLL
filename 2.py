from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv(r".\Datasets\autompg.csv")

# 'class' is mpg for some reason
df = df[['displacement', 'class']]

X = df[['displacement']].values 
y = df[['class']].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

# Linear regression
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)

y_prediction_linear = linearModel.predict(X_test)

print("Errors for Linear Regression")
print(f"MSE = {mean_squared_error(y_test, y_prediction_linear): .4f} \nR2 = {r2_score(y_test, y_prediction_linear): .4f}\n")


# Polynomial Regression
degrees = [2, 3, 4]
models = {}

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
    
# Visualization

x_plot = np.linspace(X.min(), X.max(), 500).reshape(-1,1)

plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data')


# Linear regression line
plt.plot(x_plot, linearModel.predict(x_plot), color='red', linewidth=2, label='Linear Regression')


# Polynomial regression curves
colors = ['green', 'purple', 'orange']

for degree, color in zip(degrees, colors):
    poly, model = models[degree]
    
    y_curve = model.predict(poly.transform(x_plot))
    
    plt.plot(x_plot, y_curve, color=color, label=f"Degree {degree}")


plt.xlabel("Engine Displacement")
plt.ylabel("Miles per Gallon")
plt.title("Polynomial v. Linear Regression")

plt.legend() 
plt.grid(True, alpha=0.6) 
plt.show() 


