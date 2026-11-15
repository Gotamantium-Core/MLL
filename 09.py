from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/OnlineRetail.csv', encoding='latin1')
df = df.dropna(subset=['CustomerID'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
# Aggregate per customer
customers = df.groupby('CustomerID').agg(
    NumOrders    = ('InvoiceNo',  'nunique'),
    NumItems     = ('Quantity',   'sum'),
    AvgUnitPrice = ('UnitPrice',  'mean'),
    TotalSpend   = ('TotalPrice', 'sum')
)
# Target: Low / Medium / High value segment
customers['Segment'] = pd.qcut(customers['TotalSpend'], q=3, labels=['Low', 'Medium', 'High'])
X = customers[['NumOrders', 'NumItems', 'AvgUnitPrice']]
y = customers['Segment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy = {accuracy_score(y_test, y_pred):.4f}\n")

for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

plt.figure(figsize=(24, 10))
plot_tree(model, feature_names=X.columns, class_names=['Low', 'Medium', 'High'], filled=True, fontsize=9)
plt.title("Decision Tree - Customer Segmentation (ID3 / Entropy)")
plt.show()
