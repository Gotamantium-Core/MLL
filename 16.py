import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score 

df = pd.read_csv(r".\Datasets\Mall_Customers.csv")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler() 

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)

kmeans_labels = kmeans.fit_predict(X_scaled)

print("\nK Means")
print(f"Inertia = {kmeans.inertia_: .4f} \nSillhouette Score = {silhouette_score(X_scaled, kmeans_labels): .4f}")

agg = AgglomerativeClustering(n_clusters=5)

agg_labels = agg.fit_predict(X_scaled)

print("\nAgglomerative Clustering")
print(f"Sillhouette Score = {silhouette_score(X_scaled, agg_labels): .4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=kmeans_labels, alpha=0.5,cmap="inferno")

plt.title("K Means")

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")


plt.subplot(1,2,2)
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=agg_labels, alpha=0.5, cmap="magma")


plt.title("Agglomerative (Hierarchical)")

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.tight_layout() 
plt.show() 
