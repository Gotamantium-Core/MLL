# Question 19 - Bagging vs Boosting ensembles (Titanic)
# BaggingClassifier(DecisionTree) vs AdaBoostClassifier, accuracy/precision/recall/F1.
# CRAM POINTS:
#   * Preprocess first: impute NaNs (median for numeric, mode for categorical), LabelEncode strings.
#   * Bagging = parallel trees on bootstrap samples -> reduces variance.
#   * Boosting (AdaBoost) = sequential trees, each fixing the previous errors -> reduces bias.
#   * Report the same 4 metrics for both to compare.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,accuracy_score,f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("Datasets/titanic.csv")

# Keep only useful columns and fill in missing values
df = df[["pclass","sex","age","sibsp","parch","fare","embarked","survived"]]
num_cols = ["age", "fare"]
df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
cat_cols = ["sex", "embarked"]
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

# Encode categorical strings as numbers
encoder = LabelEncoder()
df["sex"] = encoder.fit_transform(df["sex"])
df["embarked"] = encoder.fit_transform(df["embarked"])

X = df.drop(columns = ["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)

bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)

print(f"\nBagging")
print(f"Accuracy = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred):.4f}")
print(f"Recall = {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score = {f1_score(y_test, y_pred):.4f}")


# Boosting (AdaBoost)
boosting = AdaBoostClassifier(n_estimators=100, random_state=42)

boosting.fit(X_train, y_train)

y_pred = boosting.predict(X_test)

print(f"\nAdaBoost")
print(f"Accuracy = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred):.4f}")
print(f"Recall = {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score = {f1_score(y_test, y_pred):.4f}")
