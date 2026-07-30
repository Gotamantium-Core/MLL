import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv(r".\Datasets\pimadiabetes.csv")

df["class"] = df["class"].map({
    "tested_negative": 0,
    "tested_positive": 1
})


X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_no_scaling = LogisticRegression(max_iter=1000)
model_no_scaling.fit(X_train, y_train)

y_pred_no = model_no_scaling.predict(X_test)

print("Without feature scaling:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_no)} \nPrecision: {precision_score(y_test, y_pred_no)} \nRecall: {recall_score(y_test, y_pred_no)} \nF1 Score: {f1_score(y_test, y_pred_no)}")

scaler = StandardScaler() 

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LogisticRegression(max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)

print("\nWith feature scaling:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_scaled)} \nPrecision: {precision_score(y_test, y_pred_scaled)} \nRecall: {recall_score(y_test, y_pred_scaled)} \nF1 Score: {f1_score(y_test, y_pred_scaled)}")
