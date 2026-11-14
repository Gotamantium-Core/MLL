# Question 13 - MLP architectures (Wine Quality)
# MLPClassifier over several hidden-layer sizes, accuracy/precision/recall/F1 per architecture.
# CRAM POINTS:
#   * Scale features ONCE before the loop (scaling doesn't depend on the architecture).
#   * hidden_layer_sizes=(20,10) means 2 hidden layers: 20 then 10 neurons.
#   * Compare architectures by weighted precision/recall/F1 - deeper is not always better.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("Datasets/wine_quality.csv")

X = df.drop(columns=["class"])
y = df["class"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
archis = [(10,),(20,),(30,),(10,20),(20,30)]

# Scaling is architecture-independent: do it once, outside the loop
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

for arch in archis:
    model = MLPClassifier(hidden_layer_sizes=arch,activation='logistic',max_iter=2000,early_stopping=True,solver='sgd',random_state=42)
    model.fit(X_train_sc,y_train)
    y_pred = model.predict(X_test_sc)
    print(f"Performance metrics - Architecture {arch}")
    print(f"Accuracy {accuracy_score(y_test,y_pred)}")
    print(f"Precision {precision_score(y_test,y_pred,average='weighted',zero_division=0)}")
    print(f"Recall {recall_score(y_test,y_pred,average='weighted',zero_division=0)}")
    print(f"F1 Score {f1_score(y_test,y_pred,average='weighted',zero_division=0)}")
