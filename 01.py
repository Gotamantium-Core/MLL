import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df= pd.read_csv('Datasets/housing.csv')
X = df[["AveRooms"]][:1000]
y = df["MedHouseVal"][:1000]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 1. Gradient Descent ---
model_sgd = SGDRegressor(max_iter=1000, learning_rate="constant", eta0=0.01, penalty=None, random_state=42)
model_sgd.fit(X_train_scaled, y_train)
y_pred_sgd = model_sgd.predict(X_test_scaled)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

print("\nGradient Descent (SGD)\n")
print("Intercept:", model_sgd.intercept_[0])
print("Coefficient:", model_sgd.coef_[0])
print("Mean Squared Error:", mse_sgd)
print("R Squared (R2):", r2_sgd)

# --- 2. Normal Equation ---
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nNormal Equation\n")
print("Intercept:", model_lr.intercept_)
print("Coefficient:", model_lr.coef_[0])
print("Mean Squared Error:", mse_lr)
print("R Squared:", r2_lr)

plt.figure(figsize=(10, 5))
plt.scatter(X_test_scaled, y_test, alpha=0.4, s=50, color="red", label="Actual Test Data")
plt.plot(X_test_scaled, y_pred_sgd, linewidth=1.5, color="green", linestyle="--", label="SGD Prediction")
plt.plot(X_test_scaled, y_pred_lr, linewidth=1.5, color="blue", label="Normal Equation Prediction")
plt.xlabel("Average Rooms (AveRooms)")
plt.ylabel("Median House Value (MedHouseVal)")
plt.title("Model Comparison: SGD vs Normal Equation")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
