import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("Datasets/fashion_mnist.csv")
df = df.sample(10000, random_state=42) # take only 10,000 samples

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train / 255.0
X_test = X_test / 255.0

k_vals = [1, 3, 5, 7, 9]
accuracy = []
for k in k_vals:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    t0 = time.perf_counter()  # OS-independent high-res timer
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    accuracy.append(acc)

    # macro averages treat all classes equally
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"K = {k}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Time     : {elapsed:.4f}s\n")
