from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv(r".\Datasets\housing.csv")

X = df[["MedInc"]].values 
y = df[["MedHouseVal"]].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_bias = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_bias = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# Use formula theta = (X^T * X)^-1 * X^T * y
theta_normal = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

def gradientDescent(X, y, learningRate=0.01, maxIterations=5000, epsilon=1e-6):
    m = len(y)
    theta = np.zeros((X.shape[1], 1))
    costHistory = []
    
    for _ in range(maxIterations):
        predictions = X @ theta 
        errors = predictions - y
        gradient = (learningRate / m) * (X.T @ errors)
        newTheta = theta - gradient 
        
        theta_difference = np.linalg.norm(newTheta - theta)
        theta = newTheta
        
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        costHistory.append(cost)
        
        if theta_difference < epsilon:
            break
    return theta, costHistory


theta_gd, costHistory = gradientDescent(X_train_bias, y_train)
iterations = len(costHistory)

y_predicted_normal = X_test_bias @ theta_normal 
y_predicted_gd = X_test_bias @ theta_gd 


mse_normal = mean_squared_error(y_test, y_predicted_normal)
mse_gd = mean_squared_error(y_test, y_predicted_gd)

r2_normal = r2_score(y_test, y_predicted_normal)
r2_gd = r2_score(y_test, y_predicted_gd)

print(f"Gradient Descent converged after {iterations} iterations\n")

print(f"Errors for Gradient Descent: \nMSE = {mse_gd: .4f} \nR2 = {r2_gd: .4f}\n")
print(f"Errors for Normal Equation: \nMSE = {mse_normal: .4f} \nR2 = {r2_normal: .4f}\n")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sample_size = min(200, len(X_test))

plt.scatter(X_test[:sample_size], y_test[:sample_size], alpha=0.5, color='red')

xline = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
xline_scaled = scaler.transform(xline)
xline_bias = np.c_[np.ones((xline_scaled.shape[0], 1)), xline_scaled]

yline_normal = xline_bias @ theta_normal
yline_gd = xline_bias @ theta_gd

plt.plot(xline, yline_normal, color='black', linewidth=3, label='Normal Equation')
plt.plot(xline, yline_gd, color='blue', linewidth=2, linestyle=':', label='Gradient Descent')

plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")

plt.legend() 
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(costHistory, color='black', linewidth=2)
plt.xlabel("Iterations")
plt.ylabel("Cost")

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show() 
