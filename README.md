# BMP (fork)

A research fork of Block-Max Pruning with the code of the works below.

## Sparcent

[Sparcent: faster multi-vector dense retrieval with sparse centroid indices](https://doi.org/10.1007/s10115-026-02907-x)
(Knowledge and Information Systems, 2026) retrieves centroids from its sparse centroid index with
BMP, one query at a time from PyTorch code (see [sparcent](https://github.com/yfy-/sparcent)):

- Python bindings: `Searcher.search_ndarray(query, k, alpha, beta)` returns the integer document
  ids of one query as a NumPy array (low overhead for per-query calls), and results are mapped
  through the document lexicon.
- Threshold initialization from the per-term k-th scores works for any `k` (nearest stored `k`
  up to 2000, otherwise 0).

## Document reordering

[Document Reordering for Efficient Dynamic Pruning in Traditional and Learned Sparse Retrieval](https://doi.org/10.1109/ACCESS.2026.3690983)
(IEEE Access, 2026) reorders documents using past query results, e.g. by how often they were
retrieved (hit count), so that the top-k threshold rises sooner. For such indexes, the `search`
tool can change the block processing order:

- `--no-sort` processes blocks in document id order instead of by decreasing score upper bound.
- `--hybrid-threshold-factor f` processes blocks in document id order until the top-k threshold
  reaches `f` times its initial estimate, then the remaining blocks by score upper bound.

## Latency measurement

`search --num-runs n` runs each query `n` times and reports mean/p50/p95/p99 of the per-query
minimum latency; `--query-latency` also prints each query's latency.

## Usage

```bash
cargo build --release
./target/release/ciff2bmp -b 8 --compress-range -c index.ciff -o index.bmp
./target/release/search -i index.bmp -q queries.pisa -k 1000 --num-runs 3 > run.trec
```

Python bindings (`pip install ./python`, needs Rust):

```python
import bmp

bmp.ciff2bmp(ciff_file="index.ciff", output="index.bmp", bsize=8, compress_range=True)
searcher = bmp.Searcher("index.bmp")
ids = searcher.search_ndarray({"car": 3.0, "engine": 1.5}, k=10, alpha=1.0, beta=1.0)
```

The impact scores of the CIFF file must be quantized to 8 bits.

## Acknowledgements

This is a fork of [BMP](https://github.com/pisa-engine/BMP) by Antonio Mallia, Torsten Suel and
Nicola Tonellotto; all credit for Block-Max Pruning goes to them. See the original repository for
details, and cite [Faster Learned Sparse Retrieval with Block-Max Pruning](https://arxiv.org/abs/2405.01117)
(SIGIR'24).
