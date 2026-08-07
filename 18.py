import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample 
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv(r".\Datasets\iris.csv")

X = df.drop(columns=["target"])
y = df["target"]

scaler = StandardScaler() 
X = scaler.fit_transform(X)

# Bootstrapping
boot_accuracy = []
boot_f1 = [] 

for i in range(10):
    X_boot, y_boot = resample(X, y, replace=True, random_state=i)
    
    X_train, X_test, y_train, y_test = train_test_split(X_boot, y_boot, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    boot_accuracy.append(accuracy_score(y_test, y_pred))
    boot_f1.append(f1_score(y_test, y_pred, average="weighted"))

print("\nBootstrapping")
print(f"Average accuracy = {np.mean(boot_accuracy): .4f}")
print(f"Average f1 score = {np.mean(boot_f1): .4f}")

# K-Fold cross validation
model = LogisticRegression(max_iter=1000)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
f1 = cross_val_score(model, X, y, cv=kfold, scoring="f1_weighted")

print("\nK-Fold Cross Validation")
print(f"Average accuracy = {accuracy.mean(): .4f}")
print(f"Average f1 score = {f1.mean(): .4f}")
