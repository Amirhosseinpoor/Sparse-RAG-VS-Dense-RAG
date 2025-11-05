import os
import json
import faiss
import argparse
import numpy as np
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from scripts.utils_text import fuse_title_text

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

def batched_encode_texts(model, texts, batch_size=256):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        em = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
        embs.append(em)
    embs = np.vstack(embs)
    return l2_normalize(embs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default=os.path.join("datasets","scifact"))
    ap.add_argument("--out", type=str, default=os.path.join("results","dense_results.json"))
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    corpus, queries, _ = GenericDataLoader(data_folder=args.dataset_dir).load(split="test")

    doc_ids = list(corpus.keys())
    corpus_texts = [fuse_title_text(corpus[did]) for did in doc_ids]

    model = SentenceTransformer(args.model)
    doc_embs = batched_encode_texts(model, corpus_texts, args.batch_size).astype("float32")

    dim = doc_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embs)

    qids = list(queries.keys())
    qtexts = [queries[qid] for qid in qids]
    query_embs = batched_encode_texts(model, qtexts, args.batch_size).astype("float32")

    sims, idxs = index.search(query_embs, args.top_k)

    results = {}
    for row, qid in enumerate(qids):
        ids = idxs[row]
        scs = sims[row]
        results[qid] = {doc_ids[int(i)]: float(scs[j]) for j, i in enumerate(ids)}

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"Wrote dense results -> {args.out}")

if __name__ == "__main__":
    main()
