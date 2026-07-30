import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score

df = pd.read_csv(r".\Datasets\OnlineRetail.csv")

# Preprocessing
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

customer = df.groupby("CustomerID").agg({
    "Quantity": "sum",
    "UnitPrice": "mean",
    "InvoiceNo": "nunique",
    "TotalAmount": "sum"
})

# Create targets (high / low spending)
median = customer["TotalAmount"].median() 

customer["Segment"] = (customer["TotalAmount"] >= median).astype(int)

X = customer[["Quantity", "UnitPrice", "InvoiceNo"]]
y = customer["Segment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

print("Feature Importance")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature} : {importance:.4f}")