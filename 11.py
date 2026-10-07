import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("Datasets/iris.csv")

X = df[["petal length (cm)", "petal width (cm)"]].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear", C=1.0)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"\nAccuracy = {accuracy_score(y_test, pred): .4f}")

plt.figure(figsize=(8, 6))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors='k')

# Decision boundary
w = model.coef_[0]
b = model.intercept_[0]

x = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)

y_boundary = -(w[0] * x + b) / w[1]
y_margin1 = -(w[0] * x + b - 1) / w[1]
y_margin2 = -(w[0] * x + b + 1) / w[1]

plt.plot(x, y_boundary, 'k-', label='Decision Boundary')
plt.plot(x, y_margin1, 'k--', label='Margin')
plt.plot(x, y_margin2, 'k--')

# Support vectors
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100,
    facecolors='none',
    edgecolors='red',
    label='Support Vectors'
)

plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.title("Linear SVM: Iris Classification")
plt.legend()
plt.savefig("Datasets/iris_svm.png", dpi=150, bbox_inches='tight')
plt.show()
