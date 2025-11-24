import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# Carga del dataset
df = pd.read_csv("dataset_original.csv")

# Limpieza y transformación de datos
def parse_price(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace("€", "").strip()
    s = re.sub(r"[^0-9.]", "", s)
    if s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(s)
    except:
        return np.nan

df["highest_price_num"] = df["highest_price"].apply(parse_price)

df["release_year"] = df["release_date"].str[-4:].astype(float)

def parse_ps_score(x):
    try:
        return float(x)
    except:
        return np.nan

df["playstation_score_num"] = df["playstation_score"].apply(parse_ps_score)

# Selección de columnas numéricas
num_cols = [
    "highest_price_num",
    "release_year",
    "metacritic_score",
    "metacritic_rating_count",
    "metacritic_user_score",
    "metacritic_user_rating_count",
    "playstation_score_num",
    "playstation_rating_count"
]

df_num = df[num_cols].dropna().copy()

# Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Varianza explicada por los dos primeros componentes:",
      pca.explained_variance_ratio_,
      "Suma:", pca.explained_variance_ratio_.sum())

# Entrenamiento de K-Means
metrics_results = []

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    metrics_results.append({
        "k": k,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db
    })

metrics_df = pd.DataFrame(metrics_results)
print(metrics_df)

# Modelo final
best_k = 2
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_num["cluster"] = kmeans_final.fit_predict(X_scaled)

# Resumen por clúster
cluster_summary = df_num.groupby("cluster").agg(["mean", "min", "max"])
print(cluster_summary)

labels = df_num["cluster"].values

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title("Clústeres de videojuegos proyectados con PCA")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()