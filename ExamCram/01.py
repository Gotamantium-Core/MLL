# Question 01 - Linear Regression, one variable (California Housing)
# Fits GD (SGDRegressor) and Normal Equation (LinearRegression) on MedInc -> MedHouseVal.
# CRAM POINTS:
#   * Scale features with StandardScaler before GD (converges fast, same units).
#   * SGDRegressor(max_iter, learning_rate='constant', eta0=0.01, penalty=None)
#   * Normal equation = LinearRegression() (closed form, no learning rate needed).
#   * Report MSE + R2 for both, plot data points + both fitted lines.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("../Datasets/housing.csv")

X = df[["MedInc"]]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X[:1000],y[:1000],train_size=0.8,random_state=42)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_sc,y_train)
y_pred = lr.predict(X_test_sc)

gd = SGDRegressor(max_iter=10000,penalty=None,learning_rate='constant',eta0=0.01)
gd.fit(X_train_sc,y_train)
y_p_g = gd.predict(X_test_sc)

print("Performance Metrics - LR")
print(f"MSE: {mean_squared_error(y_test,y_pred)}")
print(f"R2 Score: {r2_score(y_test,y_pred)}")

print("Performance Metrics - GD (10000 iter - 0.01 learning rate)")
print(f"MSE: {mean_squared_error(y_test,y_p_g)}")
print(f"R2 Score: {r2_score(y_test,y_p_g)}")

plt.figure(figsize=(10,6))
sample_size = min(200,len(X_test))
plt.scatter(X_test_sc[:sample_size],y_test[:sample_size],color='red',label='actual data')
xline = np.linspace(float(X_test.min().iloc[0]),float(X_test.max().iloc[0]),100).reshape(-1,1) #equally distant 100 datapoints between min-max input test values
xline_sc = sc.transform(pd.DataFrame(xline))

yline_norm = lr.predict(xline_sc)
yline_gd = gd.predict(xline_sc)

plt.plot(xline,yline_norm,color="black",label="Normal Equation - Linear Regression")
plt.plot(xline,yline_gd,color="blue",label="Gradient Descent - Linear Regression")

plt.xlabel("Median Income")
plt.ylabel("House Value")

plt.grid()
plt.legend()
plt.show()
