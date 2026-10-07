from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd
import time
import matplotlib.pyplot as plt

df = pd.read_csv("Datasets/mnist.csv")
df = df.sample(5000, random_state=42)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

activations = ["logistic", "relu", "tanh"]
acs = [] ; pcs = [] ; rcs = [] ; f1s = []; ts = []
models = [] ; iters = []
for a in activations:
  model = MLPClassifier(hidden_layer_sizes=(50,), activation=a, max_iter=1000, early_stopping=True, random_state=42)

  start = time.time()
  model.fit(X_train, y_train)
  ts.append(time.time() - start)

  models.append(model)
  iters.append(model.n_iter_)
  pred = model.predict(X_test)

  acs.append(accuracy_score(y_test, pred))
  pcs.append(precision_score(y_test, pred, average='weighted'))
  rcs.append(recall_score(y_test, pred, average='weighted'))
  f1s.append(f1_score(y_test, pred, average='weighted'))

results = pd.DataFrame({
    "activation": activations,
    "accuracy":acs,
    "precision":pcs,
    "recall":rcs,
    "f1 score":f1s,
    "iterations":iters,
    "time": ts
})
numeric = ["accuracy", "precision", "recall", "f1 score", "time"]
results[numeric] = results[numeric].map(lambda x: f"{x:.4f}")
print(results.to_string(index=False))


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,6))
axes.flatten()
for i in range(3):
  axes[i].plot(models[i].loss_curve_)
  axes[i].set_title(f"Activation = {activations[i]}")
plt.tight_layout()
plt.show()
