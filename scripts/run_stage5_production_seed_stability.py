"""Confirm K-Means seed stability using the production n_init setting."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
pca = pd.read_csv(ROOT / "data/processed/stage4_selected_pca_scores.csv")
X = pca[["PC1", "PC2", "PC3", "PC4", "PC5"]].to_numpy(float)
rows = []
for k in [2, 4, 5]:
    label_sets = [
        KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        for seed in [7, 17, 27]
    ]
    scores = [
        adjusted_rand_score(label_sets[i], label_sets[j])
        for i in range(len(label_sets))
        for j in range(i + 1, len(label_sets))
    ]
    rows.append(
        {
            "config_name": f"kmeans_k{k}",
            "production_n_init": 10,
            "pairwise_ari_mean": float(np.mean(scores)),
            "pairwise_ari_min": float(np.min(scores)),
            "pairwise_ari_max": float(np.max(scores)),
        }
    )

output = pd.DataFrame(rows)
output.to_csv(
    ROOT / "outputs/tables/stage5_kmeans_production_seed_stability.csv",
    index=False,
)
print(output.to_string(index=False))
