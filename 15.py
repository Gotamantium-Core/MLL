import time 
import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv(r".\Datasets\fashion_mnist.csv")

df = df.sample(5000, random_state=42) # take 5000 samples from the dataset (optimization)

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter settings (learning rate, batch size, number of epoch)
experiments = [(0.001, 32, 100), (0.001, 64, 100), (0.01, 32, 100), (0.01, 64, 200), (0.0001, 128, 300)]

for lr, batch, epochs in experiments:
    start = time.time() 
    
    model = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=lr, batch_size=batch, max_iter=epochs, random_state=42, early_stopping=True)
    model.fit(X_train, y_train)
    
    end = time.time() 
    
    y_pred = model.predict(X_test)
    
    print(f"\nLearning Rate = {lr}")
    print(f"Batch Size = {batch}")
    print(f"Epochs = {epochs}")

    print(f"Training Time = {end-start:.2f} seconds")
    print(f"Accuracy = {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision = {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall = {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score = {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    