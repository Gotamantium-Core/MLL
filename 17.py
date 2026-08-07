import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score

df = pd.read_csv(r".\Datasets\digits.csv")

X = df.drop(columns=["target"])

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

ks = [5, 8, 10, 12, 15]

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    labels = model.fit_predict(X_scaled)
    
    print(f"\nK = {k}")
    print(f"Silhouette Score  = {silhouette_score(X_scaled, labels): .4f}")
    