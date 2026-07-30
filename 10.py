import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 


df = pd.read_csv(r".\Datasets\adult_income.csv")

X = df.drop(columns=["class"])
y = df['class']

categorical = X.select_dtypes(include=["object", "category"]).columns
numerical = X.select_dtypes(include=["int64", "float64"]).columns


preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder())
        ]), categorical)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("Logistic Regression")

print(f"Accuracy: {accuracy_score(y_test, y_pred_log)}")
print(f"Precision: {precision_score(y_test, y_pred_log, pos_label=">50K")}")
print(f"Recall: {recall_score(y_test, y_pred_log, pos_label=">50K")}")
print(f"F1 Score: {f1_score(y_test, y_pred_log, pos_label=">50K")}")


tree = DecisionTreeClassifier(criterion="entropy", random_state=42)

tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

print("\nDecision Tree")

print(f"Accuracy: {accuracy_score(y_test, y_pred_tree)}")
print(f"Precision: {precision_score(y_test, y_pred_tree, pos_label=">50K")}")
print(f"Recall: {recall_score(y_test, y_pred_tree, pos_label=">50K")}")
print(f"F1 Score: {f1_score(y_test, y_pred_tree, pos_label=">50K")}")
