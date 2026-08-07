import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv(r".\Datasets\titanic.csv")

# Only useful columns
df = df[["pclass","sex","age","sibsp","parch","fare","embarked","survived"]]

# Filing in null values
num_cols = ["age", "fare"]

num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])


cat_cols = ["sex", "embarked"]

cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

encoder = LabelEncoder()

df["sex"] = encoder.fit_transform(df["sex"])
df["embarked"] = encoder.fit_transform(df["embarked"])



X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Bagging
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


