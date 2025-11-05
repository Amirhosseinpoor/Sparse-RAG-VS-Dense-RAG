import os, pathlib
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def main():
    out_root = "datasets"
    os.makedirs(out_root, exist_ok=True)

    dataset = "scifact"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

    data_path = util.download_and_unzip(url, out_root)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"\n Downloaded SciFact to: {data_path}")
    print(f"Corpus: {len(corpus):,} | Test queries: {len(queries):,} | Qrels: {len(qrels):,}")

if __name__ == "__main__":
    main()
