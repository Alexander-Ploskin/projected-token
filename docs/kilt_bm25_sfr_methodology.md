# KILT Retrieval Methodology for Article (BM25 vs SFR-Embedding-Mistral)

## 1) Experimental setup

We evaluate retrieval quality on KILT-style OpenQA for three datasets:

- **PopQA** (`test`)
- **HotpotQA distractor** (`validation`)
- **HotpotQA fullwiki** (`test`)

Primary retrieval metrics:

- **Recall@1, Recall@5, Recall@10**
- **MRR**
- **NDCG@1, NDCG@5, NDCG@10** (reported in our SFR pipeline)

## 2) How the SFR index was built (main hyperparameters)

The dense index was built over the KILT corpus using **SFR-Embedding-Mistral** with the following configuration:

- **Backbone model**: `Salesforce/SFR-Embedding-Mistral`
- **Corpus split for indexing**: KILT `train`
- **Text field**: `text`
- **Embedding dimension**: `4096`
- **Max sequence length**: `4096`
- **Batch size (encoding)**: `8`
- **Devices**: `cuda:0`, `cuda:1`
- **Attention implementation**: `sdpa` (with fallback allowed)
- **Vector normalization**: enabled (L2)
- **Similarity metric**: inner product
- **FAISS layout**: `flat_fp16`, sharded
- **Number of vectors indexed**: `111,789,997`
- **Number of shards**: `13,974`
- **Add batch size during indexing**: `16,384`
- **Streaming shard batches per file**: `1000`
- **Metadata kept per passage**: `wikipedia_id`, `wikipedia_title`

Notes:

- The index is physically sharded due to scale.
- Retrieval uses top-k merge across all shards.

## 3) How SFR retrieval was computed at evaluation time

Query encoding logic:

- Queries are instruction-formatted:
  - `Instruct: Given a web search query, retrieve relevant passages that answer the query`
  - `Query: {question}`
- Embeddings are produced with:
  - transformer forward pass
  - last-token pooling
  - L2 normalization

Evaluation workflow:

1. Retrieve top-k passages from the SFR FAISS index.
2. Save retrieved passages (including text) into a reusable cache.
3. Recompute metrics from this cache without re-running retrieval when only relevance logic changes.

## 4) How BM25 was computed (concrete protocol)

BM25 setup in our baseline experiments:

- **BM25 formula settings**: `k1 = 1.5`, `b = 0.75`
- **BM25 implementation mode**: Lucene-style scoring
- **Tokenizer for indexing**: `bert-base-uncased`
- **Indexing scope**: full corpus (for the fullwiki setting)

Retrieval-stage parameters used in BM25 pipelines:

- **PopQA / NQ full-corpus runs**:
  - first-stage retrieval depth: `100`
  - final depth after reranking: `5`
  - reranking: cross-encoder enabled (`always`)
- **HotpotQA fullwiki runs**:
  - first-stage retrieval depth: `200`
  - final depth after reranking: `5`
  - reranking: cross-encoder enabled (`always`)
- **HotpotQA distractor runs**:
  - evaluation is over the provided distractor candidate set
  - retrieval quality is measured with top-k metrics over that candidate context set

## 5) Unified relevance logic used for comparison

To align dense and lexical baselines, we use BM25-style **in-accuracy** matching:

1. Normalize retrieved passage text and reference answer:
   - lowercase
   - remove punctuation
   - collapse repeated spaces
2. Mark a hit if normalized reference is a substring of normalized passage text.

Gold references:

- **PopQA**: object answer + possible answers
- **HotpotQA distractor/fullwiki**: answer field

If a query has empty gold references, it is excluded from the scoring denominator (`n_queries_scored`).

## 6) Current retrieval results (available now)

| Retriever | Dataset | Split | n_queries | n_queries_scored | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFR-Embedding-Mistral | PopQA | test | 14267 | 14267 | 0.440737 | 0.638887 | 0.700428 | 0.525693 | 0.440737 | 0.537601 | 0.544559 |
| SFR-Embedding-Mistral | HotpotQA distractor | validation | 7405 | 7404 | 0.335494 | 0.552269 | 0.619800 | 0.428914 | 0.335494 | 0.445850 | 0.458177 |
| SFR-Embedding-Mistral | HotpotQA fullwiki | test | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| BM25 | PopQA | test | pending* | pending* | pending* | pending* | pending* | pending* | not reported** | not reported** | not reported** |
| BM25 | HotpotQA distractor | validation | pending* | pending* | pending* | pending* | pending* | pending* | not reported** | not reported** | not reported** |
| BM25 | HotpotQA fullwiki | validation/test | pending* | pending* | pending* | pending* | pending* | pending* | not reported** | not reported** | not reported** |

\* BM25 retrieval metric JSON snapshots are not currently available in this workspace snapshot.  
\** In the BM25 evaluation pipeline, standard reported retrieval metrics are Recall@k and MRR; NDCG is not part of the default BM25 report.

## 7) Reporting notes for the paper

- Dense and lexical systems are compared under a harmonized relevance matcher (BM25-style in-accuracy).
- SFR additionally reports NDCG, which can be used to characterize ranking quality beyond first-hit metrics.
- For strict apples-to-apples comparison on Hotpot fullwiki, ensure both systems are reported on the same split (validation or test) and with clearly stated reranking settings.
