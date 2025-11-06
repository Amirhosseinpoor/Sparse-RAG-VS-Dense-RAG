# Sparse vs. Dense Retrieval on SciFact

## Project structure (brief)

```
ir-project/
├── datasets/
│   └── scifact/                 # BEIR SciFact dataset (downloaded by script)
├── results/                     # Saved retrieval runs + BM25 trials log
├── scripts/
│   ├── download_scifact.py      # Fetch & unpack SciFact
│   ├── sparse_bm25.py           # BM25 retriever (+ simple grid over k1,b,title boost)
│   ├── dense_faiss.py           # Dense retriever with Sentence-Transformers + FAISS
│   └── utils_text.py            # Tokenization & text utilities
├── evaluation.py                # Standardized evaluation wrapper (BEIR)
├── README.md
└── requirements.txt
```

## Setup & how to run (step-by-step)

1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate         # on Windows: venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download the SciFact dataset

```bash
python scripts/download_scifact.py
# Dataset will appear under datasets/scifact/
```

4. Run the sparse retriever (BM25)

```bash
python scripts/sparse_bm25.py --dataset_dir datasets/scifact --objective ndcg@10
# Writes: results/sparse_results.json  (and logs trials to results/bm25_grid_trials.tsv)
```

**Note:** We fine-tuned BM25 via a small grid search. The best configuration used **k1 = 0.95** and **b = 0.95**, which yielded the best accuracy on this setup.

5. Run the dense retriever (Sentence-Transformers + FAISS)

```bash
python -m scripts.dense_faiss --dataset_dir datasets/scifact --out results/dense_results.json
```

6. Evaluate both runs with the standardized script

```bash
python evaluation.py datasets/scifact results/sparse_results.json
python evaluation.py datasets/scifact results/dense_results.json
```

## Results (test split)

| Metric     | BM25 (Sparse) | Dense (FAISS) |
| ---------- | ------------- | ------------- |
| NDCG@10    | **0.66563**   | 0.64840       |
| NDCG@100   | **0.68801**   | 0.67833       |
| MAP@10     | **0.62297**   | 0.59895       |
| MAP@100    | **0.62822**   | 0.60547       |
| Recall@10  | 0.78233       | **0.78833**   |
| Recall@100 | 0.87972       | **0.92500**   |
| P@10       | 0.08600       | **0.08900**   |
| P@100      | 0.00987       | **0.01053**   |

**Which retriever performed better?**
BM25 achieved higher ranking quality (NDCG and MAP). The dense retriever achieved higher coverage (Recall@100) and slightly better precision at fixed cutoffs.

**Why?**
SciFact contains scientific claims where exact and near-exact term matches (especially titles) are very informative; BM25 benefits from this lexical signal, improving ranking metrics. Dense embeddings broaden semantic matching, which increases recall (more relevant docs surfaced) but can include semantically related yet less directly relevant items, softening NDCG/MAP.

**Trade-offs observed**

* **Speed (build time):** BM25 indexes quickly on CPU. Dense requires embedding the corpus and queries and benefits from a GPU.
* **Memory:** BM25 postings are compact. Dense stores float embeddings and a FAISS index.
* **Retrieval quality:** BM25 led on **NDCG/MAP**; Dense led on **Recall** and edged **P@k**—useful when downstream rerankers/generators can re-score wider candidate sets.
