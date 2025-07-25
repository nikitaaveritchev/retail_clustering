from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class KMeansResult:
    model: KMeans
    labels: pd.Series
    silhouette: float


def evaluate_clusters(data: pd.DataFrame, k_values: Iterable[int], random_state: int = 42) -> dict[int, float]:
    scores: dict[int, float] = {}
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(data)
        score = silhouette_score(data, labels)
        scores[k] = score
    return scores


def run_kmeans(
    data: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> KMeansResult:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(scaled)
    score = silhouette_score(scaled, labels)
    return KMeansResult(km, pd.Series(labels, index=data.index), score)
