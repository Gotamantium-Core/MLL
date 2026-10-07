from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA # for visualization

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("Datasets/digits.csv")

X = df.drop(columns=["target"])
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)

ks = [4, 5, 7, 9, 11, 13]
silhs = [] ; labs = [] ; ins = []
for k in ks:
  model = KMeans(n_clusters=k, random_state=42, n_init=10)
  labels = model.fit_predict(X_scaled)
  labs.append(labels)
  ins.append(model.inertia_)
  silhs.append(silhouette_score(X_scaled, labels))

results = pd.DataFrame({
    "K_Value":ks,
    "Inertia":ins,
    "Silhouette_Score":silhs
})
numeric = ["Inertia", "Silhouette_Score"]
results[numeric] = results[numeric].map(lambda x: f"{x:.4f}")
print(results.to_string(index=False))

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,8))
axes = axes.flatten()

for i in range(6):
  axes[i].scatter(X_reduced[:,0], X_reduced[:,1], c=labs[i], cmap="inferno", alpha=0.5, s=15)
  axes[i].set_title(f"K={ks[i]}")
plt.tight_layout()
plt.show()
