import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r".\Datasets\fashion_mnist.csv")
df = df.sample(10000, random_state=42) # take only 10,000 samples 

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train / 255.0
X_test = X_test / 255.0

k_vals = [1, 3, 5, 7, 9]
accuracy = [] 
for k in k_vals:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracy.append(acc)
    
    print(f"K = {k} \nAccuracy: {acc: .4f}\n")
