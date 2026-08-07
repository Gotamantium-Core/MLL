import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

df = pd.read_csv(r".\Datasets\wine_quality.csv")

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

architectures = [(10,), (20,), (30,), (20,10), (30,20)]

for arch in architectures:
    model = MLPClassifier(hidden_layer_sizes=arch, max_iter=2000, early_stopping=True, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\nArchitecture {arch}: ")
    
    print(f"Accuracy = {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision = {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall = {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score = {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
