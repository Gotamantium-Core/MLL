import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 

df = pd.read_csv(r".\Datasets\iris.csv")

X = df.iloc[:, :2].values # take the two features for visualization
y = (df["target"] == 0).astype(int) # setosa = 1, others=0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler() 

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear", C=1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")



plt.figure(figsize=(8,6))

xmin, xmax = X_train[:,0].min() - 1, X_train[:,0].max() + 1 
ymin, ymax = X_train[:,1].min() - 1, X_train[:,1].max() + 1 

xx, yy = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, alpha=0.3)

colors = ["midnightblue" if label == 0 else "forestgreen" for label in y_train]
plt.scatter(X_train[:,0], X_train[:,1], c=colors, alpha=0.5)

# Draw decision boundary and margins
ax = plt.gca() 

xlim = ax.get_xlim()
ylim = ax.get_ylim() 

XX = np.linspace(xlim[0], xlim[1], 30)
YY = np.linspace(ylim[0], ylim[1], 30)

YY, XX = np.meshgrid(YY, XX)

xy = np.vstack([XX.ravel(), YY.ravel()]).T 

Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.8, linestyles=["--", "-", "--"])

ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=100, facecolors='none', edgecolors='crimson')

plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])

plt.show() 

