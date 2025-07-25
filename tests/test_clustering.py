import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from retail_clustering.clustering import evaluate_clusters, run_kmeans


def test_run_kmeans():
    df = pd.DataFrame({
        "Recency": [10, 20, 30, 10, 20, 30],
        "Frequency": [1, 2, 1, 1, 2, 1],
        "Monetary": [100, 200, 150, 90, 210, 160],
    })
    scores = evaluate_clusters(df, range(2, 4))
    assert len(scores) == 2
    res = run_kmeans(df, 2)
    assert res.labels.nunique() == 2
