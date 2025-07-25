# Retail Data Clustering Project

## Description
This project focuses on clustering retail data to segment customers based on their purchasing behavior. The analysis is built around **RFM analysis** (Recency, Frequency, Monetary value) combined with **K-Means** clustering. The code is adapted from the tutorial [Clustering Retail Data](https://www.youtube.com/watch?v=afPJeQuVeuY&t=2000s) and extended with a reusable Python package and command line interface.

## Usage
1. Install dependencies (preferably in a virtual environment):
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
2. Run the pipeline on your retail Excel file:
   ```bash
   python -m retail_clustering.pipeline data/online_retail_II.xlsx --out results.csv
   ```
   The script will automatically determine the optimal number of clusters using the silhouette score.

## Development
Unit tests are provided to validate the RFM calculations and clustering logic:
```bash
pytest -q
```
