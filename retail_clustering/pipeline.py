from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data_loader import load_excel
from .rfm import calculate_rfm
from .clustering import evaluate_clusters, run_kmeans


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retail clustering pipeline")
    parser.add_argument("path", help="Path to Excel file")
    parser.add_argument("--sheet", type=int, default=0, help="Excel sheet index")
    parser.add_argument("--clusters", type=int, help="Number of clusters")
    parser.add_argument("--out", type=Path, help="Where to write results as CSV")
    args = parser.parse_args(argv)

    df = load_excel(args.path, sheet_name=args.sheet)
    rfm = calculate_rfm(df)

    if args.clusters is None:
        scores = evaluate_clusters(rfm, range(2, 9))
        args.clusters = max(scores, key=scores.get)

    result = run_kmeans(rfm, args.clusters)
    rfm["Cluster"] = result.labels

    if args.out:
        rfm.to_csv(args.out, index=True)
    else:
        print(rfm.head())
        print("Silhouette:", result.silhouette)


if __name__ == "__main__":
    main()
