Суммаризирую последнюю конфигурацию из нашего диалога (stage-b-query-distill-bge-base-flatten-localbeir-bm25hard-val250-soft):

1. Модель учитель
BAAI/bge-base-en-v1.5 — генерирует teacher-эмбеддинги для query и document с раздельными prefix'ами:

Document: BGE_DOCUMENT_PREFIX + text

Query: BGE_QUERY_PREFIX + text

Размерность: 768

2. Архитектура модели
Базовая модель: naver/oscar-qwen2-7B (hidden_dim = 3584, 8 MEM-токенов)

Проектор:

text
flatten(8 × 3584) = 28672
        ↓  Linear + GeLU
      8192
        ↓  Linear
       768  → L2-normalize
projector_type: mem, num_layers: 2, pooler: flatten

Один проектор для query и document (shared weights)

dropout: 0.0

Состав train mixture

Датасет	Пар	Hard negatives	Домен
MSMARCO	~500K	BGE dense (rank 10–100)	General web QA
nfcorpus-train	~3.2K	BM25 + BGE hybrid	Biomedical
fiqa-train	~14K	BGE dense (BM25 плохо работает для finance)	Finance
arguana	~1.4K	BGE dense	Argumentation
quora-train	~15K	BGE dense	Paraphrase/QA
scifact-train	~0.8K	BM25 + BGE hybrid	Science
Итого: ~534K пар

Стратегия hard negatives по источнику

text
MSMARCO:     готовые BGE negatives (BAAI/MSMARCO-with-hard-negatives)
             → скачать напрямую, не генерировать

fiqa + quora + arguana:  BGE dense mining
             index = faiss(BGE.encode(corpus))
             negs = top-200 → rank 10–100, score-based filter
             [pos_score - 0.30 < neg_score < pos_score - 0.05]

nfcorpus + scifact:  BM25 ∪ BGE hybrid
             bm25_negs = BM25_top100[10:60]
             bge_negs  = BGE_top100[10:60]
             candidates = bm25_negs | bge_negs
             # Cross-encoder filter: убрать false negatives (CE score > 0.5)
             hard_negs = [d for d in candidates if CE(query, d) < 0.5]

Итоговая конфигурация датасета
Состав train mixture

Датасет	Пар	Hard negatives	Домен
MSMARCO	~500K	BGE dense (rank 10–100)	General web QA
nfcorpus-train	~3.2K	BM25 + BGE hybrid	Biomedical
fiqa-train	~14K	BGE dense (BM25 плохо работает для finance)	Finance
arguana	~1.4K	BGE dense	Argumentation
quora-train	~15K	BGE dense	Paraphrase/QA
scifact-train	~0.8K	BM25 + BGE hybrid	Science
Итого: ~534K пар

Стратегия hard negatives по источнику

text
MSMARCO:     готовые BGE negatives (BAAI/MSMARCO-with-hard-negatives)
             → скачать напрямую, не генерировать

fiqa + quora + arguana:  BGE dense mining
             index = faiss(BGE.encode(corpus))
             negs = top-200 → rank 10–100, score-based filter
             [pos_score - 0.30 < neg_score < pos_score - 0.05]

nfcorpus + scifact:  BM25 ∪ BGE hybrid
             bm25_negs = BM25_top100[10:60]
             bge_negs  = BGE_top100[10:60]
             candidates = bm25_negs | bge_negs
             # Cross-encoder filter: убрать false negatives (CE score > 0.5)
             hard_negs = [d for d in candidates if CE(query, d) < 0.5]
Валидация

text
Train H5:
  msmarco-hard.h5          ← основной, 500K
  nfcorpus-train-hard.h5
  fiqa-train-hard.h5
  arguana-hard.h5
  quora-train-hard.h5
  scifact-train-hard.h5    ← только train split!

Val proxy (model selection, BEIR probe):
  scifact-dev queries       ← held-out, никогда не в train

Final test (один раз):
  BEIR3: scifact-test + nfcorpus-test + fiqa-test

Итоговая конфигурация датасета
Состав train mixture

Датасет	Пар	Hard negatives	Домен
MSMARCO	~500K	BGE dense (rank 10–100)	General web QA
nfcorpus-train	~3.2K	BM25 + BGE hybrid	Biomedical
fiqa-train	~14K	BGE dense (BM25 плохо работает для finance)	Finance
arguana	~1.4K	BGE dense	Argumentation
quora-train	~15K	BGE dense	Paraphrase/QA
scifact-train	~0.8K	BM25 + BGE hybrid	Science
Итого: ~534K пар

Стратегия hard negatives по источнику

text
MSMARCO:     готовые BGE negatives (BAAI/MSMARCO-with-hard-negatives)
             → скачать напрямую, не генерировать

fiqa + quora + arguana:  BGE dense mining
             index = faiss(BGE.encode(corpus))
             negs = top-200 → rank 10–100, score-based filter
             [pos_score - 0.30 < neg_score < pos_score - 0.05]

nfcorpus + scifact:  BM25 ∪ BGE hybrid
             bm25_negs = BM25_top100[10:60]
             bge_negs  = BGE_top100[10:60]
             candidates = bm25_negs | bge_negs
             # Cross-encoder filter: убрать false negatives (CE score > 0.5)
             hard_negs = [d for d in candidates if CE(query, d) < 0.5]
Валидация

text
Train H5:
  msmarco-hard.h5          ← основной, 500K
  nfcorpus-train-hard.h5
  fiqa-train-hard.h5
  arguana-hard.h5
  quora-train-hard.h5
  scifact-train-hard.h5    ← только train split!

Val proxy (model selection, BEIR probe):
  scifact-dev queries       ← held-out, никогда не в train

Final test (один раз):
  BEIR3: scifact-test + nfcorpus-test + fiqa-test
Сэмплирование при обучении

MSMARCO в 30× больше остальных — без взвешивания он задавит остальные домены:

text
dataset_sampling_weights:
  msmarco: 0.50        # 50% батча
  fiqa: 0.15
  quora: 0.15
  nfcorpus: 0.08
  scifact: 0.07
  arguana: 0.05

recipe: query_distill
teacher: BAAI/bge-base-en-v1.5
pooler: flatten
projector_hidden_dim: 8192
num_layers: 2
embed_dim: 768
batch_size: 64          # увеличить с 16 — больше in-batch negatives
lr: 2.0e-5
warmup_steps: 1000      # дольше warmup под больший датасет
max_steps: 20000        # ~2 эпохи на полном миксе
min_lr: 1.0e-6
scheduler: cosine
query_mse_weight: 1.0
doc_mse_weight: 1.0
negative_mse_weight: 0.5
ranking_weight: 0.1
margin: 0.10            # поднять с 0.05 — hard negatives теперь реально сложные
early_stopping_patience: 3
selection_metric: proxy_ndcg@10


Валидация должна быть каждые 250 шагов! Проба на ms-marco и на hard negatives + MRR на валидации тоже считать