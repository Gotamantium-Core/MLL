# Question 11 - Linear SVM binary classification (Iris: Setosa vs Non-Setosa)
# SVC(kernel='linear') on scaled petal features, then draw boundary + both margin lines.
# CRAM POINTS:
#   * Binary-ify: Setosa -> 0, everything else -> 1 (Y.map).
#   * Boundary: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x + b)/w2.
#   * Margins: same line with b shifted by +-1 (w.x + b = +1 and w.x + b = -1).
#   * Support vectors = points on/inside the margins; they alone define w, b.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv("../Datasets/iris.csv")
X = df[["petal length (cm)","petal width (cm)"]]
# Setosa -> 0, Non-Setosa -> 1
Y = df["target"].map({0:0,1:1,2:1})

X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.8,random_state=42)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

model = SVC(C=1.0,kernel='linear')
model.fit(X_train_sc,y_train)
y_pred = model.predict(X_test_sc)

print(f"Accuracy: {accuracy_score(y_test,y_pred)}")

plt.scatter(X_test_sc[:,0],X_test_sc[:,1],c=y_pred)
print(model.coef_)
w = model.coef_[0]
b = model.intercept_[0]

x = np.linspace(float(X_test_sc.min()),float(X_test_sc.max()),100)

svm=-(w[0]*x+b)/w[1]
svm_a=-(w[0]*x+b-1)/w[1]
svm_b=-(w[0]*x+b+1)/w[1]

plt.plot(x,svm,color='black',label="SVM")
# class +1 = Non-Setosa sits above the +1 margin, class 0 = Setosa below the -1 margin
plt.plot(x,svm_a,color="red",label = "Non Setosa margin (+1)")
plt.plot(x,svm_b,color="blue",label="Setosa margin (-1)")

plt.legend()
plt.grid(True)

plt.show()
