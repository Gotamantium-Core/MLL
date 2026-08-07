import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 

df = pd.read_csv(r".\Datasets\boston_housing.csv")

X = df.drop(columns=["MEDV"])
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

degrees = [1, 2, 3, 4, 5]

training_error = []
validation_error = [] 

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression() 
    model.fit(X_train_poly, y_train)
    
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    
    training_error.append(mean_squared_error(y_train, train_pred))
    validation_error.append(mean_squared_error(y_test, test_pred))
    
    print(f"\nDegree = {degree}")
    print(f"Training error = {training_error[-1]: .4f}")
    print(f"Validation error = {validation_error[-1]: .4f}")
    
plt.figure(figsize=(7,5))

plt.plot(degrees, training_error, marker='o', label='Training Error', color="deeppink")

plt.plot(degrees, validation_error, marker='s', label='Validation Error', color="slateblue")

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Tradeoff")

plt.legend() 
plt.grid(True)
plt.show() 
    