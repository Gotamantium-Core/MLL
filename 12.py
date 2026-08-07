import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score 

df = pd.read_csv(r".\Datasets\fashion_mnist.csv")

df = df.sample(5000, random_state=42) # take 5000 samples from the dataset (optimization)

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Linear
linear = SVC(kernel="linear")
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)

print("\nLinear SVM")
print(f"Accuracy = {accuracy_score(y_test, y_pred): .4f}")
print(f"Precision = {precision_score(y_test, y_pred, average="weighted"): .4f}")
print(f"Recall = {recall_score(y_test, y_pred, average="weighted"): .4f}")
print(f"F1 Score = {f1_score(y_test, y_pred, average="weighted"): .4f}")

# Polynomial (degree = 3)
poly = SVC(kernel="poly", degree=3)
poly.fit(X_train, y_train)
y_pred = poly.predict(X_test)

print("\nPolynomial SVM (degree = 3)")
print(f"Accuracy = {accuracy_score(y_test, y_pred): .4f}")
print(f"Precision = {precision_score(y_test, y_pred, average="weighted"): .4f}")
print(f"Recall = {recall_score(y_test, y_pred, average="weighted"): .4f}")
print(f"F1 Score = {f1_score(y_test, y_pred, average="weighted"): .4f}")


# RBF
rbf = SVC(kernel="rbf")
rbf.fit(X_train, y_train)
y_pred = rbf.predict(X_test)

print("\nRBF SVM")
print(f"Accuracy = {accuracy_score(y_test, y_pred): .4f}")
print(f"Precision = {precision_score(y_test, y_pred, average="weighted"): .4f}")
print(f"Recall = {recall_score(y_test, y_pred, average="weighted"): .4f}")
print(f"F1 Score = {f1_score(y_test, y_pred, average="weighted"): .4f}")



