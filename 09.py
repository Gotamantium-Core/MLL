from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'Datasets/iris.csv')

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy = {accuracy_score(y_test, y_pred):.4f}\n")

for feature, importance in zip(X.columns, model.feature_importances_):
  print(f"{feature}: {importance: .4f}")

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, filled=True, fontsize=10)
plt.show()
