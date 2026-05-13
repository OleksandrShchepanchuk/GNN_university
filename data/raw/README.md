# Raw data — SNAP Reddit Hyperlink Network

Three files from <https://snap.stanford.edu/data/soc-RedditHyperlinks.html>:

| File                                       | Description                                                              |
|--------------------------------------------|--------------------------------------------------------------------------|
| `soc-redditHyperlinks-body.tsv`            | Hyperlinks extracted from post **bodies** (264k edges, with timestamps + sentiment label + 86-D text properties). |
| `soc-redditHyperlinks-title.tsv`           | Hyperlinks extracted from post **titles** (572k edges, same schema).     |
| `web-redditEmbeddings-subreddits.csv`      | 300-D LIWC-derived subreddit embeddings (one row per subreddit).         |

Each TSV row has columns:

```
SOURCE_SUBREDDIT  TARGET_SUBREDDIT  POST_ID  TIMESTAMP  POST_LABEL  POST_PROPERTIES
```

- `POST_LABEL`: `-1` (negative) or `+1` (neutral/positive). The project remaps
  `-1 → 0`, `+1 → 1` for the edge sign classification task.
- `POST_PROPERTIES`: comma-separated 86-D vector of LIWC/text features.

## Download

Recommended (after `make install`):

```bash
python scripts/prepare_data.py
```

Or manually with `curl`:

```bash
cd data/raw
curl -L -O https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv
curl -L -O https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv
curl -L -O https://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv
```

The raw files are **not** committed (see `.gitignore`).

## Task reminder

> Edge sign classification on observed hyperlinks. We never sample non-edges;
> `label == 0` always means "negative sentiment", never "no edge".
