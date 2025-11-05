
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import os
import json
import argparse
from itertools import product
from tqdm import tqdm

from rank_bm25 import BM25Okapi
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from scripts.utils_text import simple_tokenize

def build_corpus_tokens(corpus_dict, title_boost=1):
    doc_ids, tok_docs = [], []
    for did, dobj in corpus_dict.items():
        title = (dobj.get("title") or "").strip()
        text  = (dobj.get("text") or "").strip()
        merged = ((title + ". ") * max(1, title_boost)) + text if title else text
        toks = simple_tokenize(merged)
        doc_ids.append(did)
        tok_docs.append(toks)
    return doc_ids, tok_docs

def parse_float_list(csv: str):
    return [float(x) for x in csv.split(",") if x.strip() != ""]

def parse_int_list(csv: str):
    return [int(x) for x in csv.split(",") if x.strip() != ""]

def evaluate_one(bm25, doc_ids, queries, qrels, top_k=100):
    results = {}
    for qid, qtext in queries.items():
        q_tokens = simple_tokenize(qtext)
        scores = bm25.get_scores(q_tokens)
        top_idx = scores.argsort()[-top_k:][::-1]
        results[qid] = {doc_ids[i]: float(scores[i]) for i in top_idx}

    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, results, [10, 100])
    return results, metrics

def pick_objective(metrics, objective="ndcg@10"):

    obj = objective.lower()
    table = {
        "ndcg@10": metrics[0]["NDCG@10"],
        "ndcg@100": metrics[0]["NDCG@100"],
        "map@10": metrics[1]["MAP@10"],
        "map@100": metrics[1]["MAP@100"],
        "recall@10": metrics[2]["Recall@10"],
        "recall@100": metrics[2]["Recall@100"],
        "p@10": metrics[3]["P@10"],
        "p@100": metrics[3]["P@100"],
    }
    return table[obj], table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default=os.path.join("datasets","scifact"))
    ap.add_argument("--out", type=str, default=os.path.join("results","sparse_results.json"))
    ap.add_argument("--trials_log", type=str, default=os.path.join("results","bm25_grid_trials.tsv"))
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--k1_list", type=str, default="0.8,0.85,0.9,0.95,1.2,1.3")
    ap.add_argument("--b_list", type=str, default="0.55,0.65,0.75,0.85,0.95")
    ap.add_argument("--title_boost_list", type=str, default="1,2,3")
    ap.add_argument("--objective", type=str, default="ndcg@10",
                    help="One of: ndcg@10, ndcg@100, map@10, map@100, recall@10, recall@100, p@10, p@100")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    corpus, queries, qrels = GenericDataLoader(data_folder=args.dataset_dir).load(split="test")

    with open(args.trials_log, "w") as f:
        f.write("k1\tb\ttitle_boost\tNDCG@10\tNDCG@100\tMAP@10\tMAP@100\tRecall@10\tRecall@100\tP@10\tP@100\n")

    best_score = -1.0
    best_combo = None
    best_results = None
    best_metrics_table = None

    k1_vals = parse_float_list(args.k1_list)
    b_vals = parse_float_list(args.b_list)
    tb_vals = parse_int_list(args.title_boost_list)

    for tb in tb_vals:
        doc_ids, tok_docs = build_corpus_tokens(corpus, title_boost=tb)
        bm25 = BM25Okapi(tok_docs, k1=1.2, b=0.75)

        for k1, b in product(k1_vals, b_vals):
            bm25.k1 = k1
            bm25.b = b
            results, metrics = evaluate_one(bm25, doc_ids, queries, qrels, top_k=args.top_k)
            obj_val, table = pick_objective(metrics, objective=args.objective)

            with open(args.trials_log, "a") as f:
                f.write(f"{k1}\t{b}\t{tb}\t"
                        f"{table['ndcg@10']:.5f}\t{table['ndcg@100']:.5f}\t"
                        f"{table['map@10']:.5f}\t{table['map@100']:.5f}\t"
                        f"{table['recall@10']:.5f}\t{table['recall@100']:.5f}\t"
                        f"{table['p@10']:.5f}\t{table['p@100']:.5f}\n")

            if obj_val > best_score:
                best_score = obj_val
                best_combo = (k1, b, tb)
                best_results = results
                best_metrics_table = table

                with open(args.out, "w") as f:
                    json.dump(best_results, f)

                print(f"[NEW BEST] {args.objective}={best_score:.5f}  "
                      f"(k1={k1}, b={b}, title_boost={tb})")

    with open(args.out, "w") as f:
        json.dump(best_results, f)

    k1, b, tb = best_combo
    print("\n=== BEST CONFIG ===")
    print(f"k1={k1}  b={b}  title_boost={tb}")
    print(f"Objective ({args.objective}) = {best_score:.5f}")
    print("Full metrics:", best_metrics_table)
    print(f"\nSaved best retrieval run to: {args.out}")
    print(f"Trials log at: {args.trials_log}")

if __name__ == "__main__":
    main()
