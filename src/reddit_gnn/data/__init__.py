"""Data loading, preprocessing, feature building, splits, and PyG Dataset.

The pipeline (to be implemented):
    download  ->  load (TSV/CSV)
              ->  preprocess (label remap, normalize subreddit names, dedupe)
              ->  features (LIWC embeddings + engineered edge features)
              ->  splits (temporal by default)
              ->  pyg_dataset (in-memory Data with edge_index, edge_attr, y)
"""
