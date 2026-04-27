**Да, понял вашу задачу полностью.** Вы проанализировали OSCAR (из arXiv:2504.07109v2) и увидели, что query-independent вариант его эмбеддингов (т.е. сжатие документа без вопроса) плохо подходит для классического dense retrieval / поиска по индексу (FAISS/HNSW и т.д.). При этом вам нужна модель, которая:

- **Сжимает контекст** только документа (query-independent compression, как в OSCAR или ICAE) — превращает длинный документ в компактное представление (несколько эмбеддинг-токенов / memory slots).
- **Одновременно служит энкодером для IR** — эти сжатые эмбеддинги можно индексировать и эффективно искать по косинусному сходству (или late-interaction) с query-эмбеддингом.

Это реально сделать. Давайте сначала кратко разберём, как сейчас строят embedding-модели для IR, а потом — конкретный план обучения вашей гибридной модели «OSCAR-подобный compressor + IR-encoder».

### 1. Как вообще делают embedding-модели для Information Retrieval (dense retrieval)

Большинство современных моделей — это **bi-encoder** (dual-encoder) архитектуры.

- **Query-encoder** и **Document-encoder** (часто один и тот же backbone, но иногда асимметричный).
- Документы кодируются **query-independent** (один раз при индексации) → фиксированный вектор (или несколько векторов).
- Query кодируется на лету → ищется по индексу (ANN).
- Основной loss — **contrastive** (InfoNCE / SimCSE / E5-style):

\[
\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\sum_{d^-} \exp(\text{sim}(q, d^-)/\tau)}
\]

где \(\text{sim}\) — косинусное сходство, \(\tau\) — temperature, \(d^+\) — positive документ, \(d^-\) — hard negatives.

**Ключевые практики обучения (state-of-the-art 2024–2026):**
- **Данные**: MS MARCO, Natural Questions + синтетика от LLM (HyDE, LLM-as-a-judge для hard negatives).
- **Hard negatives mining**: BM25 → cross-encoder (DeBERTa) → top-k самых «похожих, но неправильных».
- **Pre-training**:
  - Unsupervised contrastive (cropping spans внутри одного документа).
  - Span-corruption / auto-encoding (как в T5).
- **Для длинных документов**:
  - Chunking + pooling (mean / CLS / last token).
  - Long-context backbones (RoPE + NTK, M2-BERT и т.д.).
  - Иногда multi-vector (ColBERT-style) — лучше качество, но тяжелее индекс.
- **Популярные модели**: BGE, E5, Snowflake, GTE — все bi-encoder + massive data augmentation.

Главное отличие от OSCAR: в OSCAR сжатие **query-dependent** (комpressor видит query + doc и делает «мягкое» сжатие под конкретный вопрос). Поэтому query-independent эмбеддинги OSCAR «размытые» и плохо работают для чистого cosine-search.

### 2. Как обучить свою модель: «OSCAR-style compression + IR-encoder»

Идея — взять архитектуру **In-context Autoencoder (ICAE)** / OSCAR-N-Layers и добавить к ней retrieval-objective. Получится query-independent compressor, который:
- Сжимает doc → 8–16 memory slots (эмбеддингов) вместо 128+ токенов (16×+ compression).
- Эти slots можно **pool-ить в один вектор** (mean-pooling) или использовать как multi-vector.
- Модель отлично индексируется и ищется.

#### Предлагаемая архитектура (на базе OSCAR / ICAE)
- **Backbone**: Llama-3.2-1B / Mistral-7B / Qwen2-1B (как в OSCAR-llama) или просто первые N слоёв большого LLM (OSCAR-N-Layers).
- **Compressor**:
  - Input: `[DOC] + document tokens + [MEM_1] ... [MEM_K]` (K=8–16 learnable memory tokens).
  - Выход: hidden states последних слоёв только на [MEM] токенах → это и есть сжатые эмбеддинги (как soft prompts в OSCAR).
- **Query-encoder**: тот же backbone (queries короткие, compression не нужен) или лёгкая отдельная голова.
- **Pooling** (на выбор):
  - Mean-pool всех K эмбеддингов → один 4096-dim вектор (просто и быстро).
  - Или late-interaction (ColBERT) — оставляем K векторов.

#### Этапы обучения (2–3 стадии, 1–10 GPU-дней на 1B-модели)

1. **Stage 0: Pre-training compressor (как в OSCAR-llama / ICAE)**  
   - Задачи:
     - Auto-encoding: восстановить оригинальный документ из memory slots (decoder = тот же LLM).
     - Text-continuation / language modeling на сжатом представлении.
   - Данные: Wikipedia + MS MARCO passages (миллиарды токенов).
   - Цель: memory slots должны **содержать всю полезную информацию** документа (как в ICAE).

2. **Stage 1: Unsupervised contrastive pre-training**  
   - Внутри одного документа создаём positive/negative spans (cropping).
   - Loss: InfoNCE на сжатых memory slots.

3. **Stage 2: Supervised retrieval fine-tuning (главная стадия)**  
   - Данные: MS MARCO + BEIR + ваши доменные данные (если есть).
   - Для каждого query:
     - Query-encoder → query embedding.
     - Doc-compressor → K memory slots → pool в doc embedding.
     - Hard negatives: mining на лету (или pre-mined).
   - Loss: InfoNCE + optionally distillation от teacher (полный doc → cross-encoder).
   - Опционально: добавить reranking-голову как в OSCAR ([RR] token), если хотите one-model-for-all.

4. **(Опционально) Stage 3: End-to-end RAG distillation**  
   - Как в OSCAR: distill от teacher-LLM (Mistral-7B), который видит **полный** документ, но ваша модель использует только compressed version. Это сделает эмбеддинги ещё лучше для downstream RAG.

#### Практические советы по реализации
- **Библиотеки**: Hugging Face + FlagEmbedding (BAAI) или Sentence-Transformers + LoRA (PEFT).
- **Compression ratio**: начинайте с 16× (128 токенов → 8 slots), как в OSCAR.
- **Индексация**: 
  - Mean-pool → обычный FAISS (IndexFlatIP или HNSW).
  - Multi-vector → ColBERTv2-style index или PLAID.
- **Оценка**:
  - MTEB / BEIR (retrieval metrics).
  - Needle-in-haystack на сжатых эмбеддингах.
  - Сравните query-independent OSCAR (baseline) vs ваша модель.
- **Потенциальные улучшения**:
  - Асимметрия: doc-compressor может быть тяжелее query-encoder (документы индексируются один раз).
  - Synthetic data: LLM генерит hypothetical documents (HyDE) для augmentation.

Такая модель будет **сжимать** контекст как OSCAR/ICAE (query-independent, soft compression) и **отлично работать** как IR-encoder, потому что мы явно оптимизируем под retrieval loss. Query-independent OSCAR не работал именно потому, что его не учили на contrastive retrieval — мы это исправим.

Если хотите — могу набросать:
- полный config для LoRA + training script,
- или конкретный промпт для синтетики данных,
- или сравнительную таблицу с существующими моделями (BGE, E5, ICAE и т.д.).

Готов углубиться в любой этап! Что дальше?

**Отлично, давай сделаем максимально простой и быстрый эксперимент.**  

Идея понятна:  
- Берём **замороженный** OSCAR compressor (query-independent режим).  
- Извлекаем hidden states MEM-токенов (как в бумаге: последние l=8 токенов).  
- Кидаем их в **новый лёгкий проектор** (MLP), который учим **только** под cosine similarity retrieval.  

Всё остальное (backbone OSCAR) — frozen. Обучение займёт часы-дни на одной GPU, а не недели.

### 1. Что именно будет проектором (самый простой вариант)
```python
class MEMProjector(nn.Module):
    def __init__(self, hidden_dim=2048, embed_dim=768):  # hidden_dim — из OSCAR (Llama-1B ~2048–4096)
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # или сразу в embed_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)  # полезно для cosine
        )
    
    def forward(self, mem_hiddens):  # mem_hiddens: [batch, l, hidden_dim] где l=8
        # Варианты pooling:
        pooled = mem_hiddens.mean(dim=1)          # простой mean (рекомендую сначала)
        # pooled = mem_hiddens[:, -1]             # или last MEM
        # pooled = torch.cat([mem_hiddens.mean(1), mem_hiddens.max(1)[0]], dim=-1)  # concat
        return self.mlp(pooled)                   # → [batch, embed_dim]
```

Это ровно то, что было в OSCAR-llama (у них уже есть 2-layer dense для генератора), но теперь мы делаем отдельный проектор **под IR**.

### 2. Полный пайплайн обучения проектора (шаг за шагом)

#### Шаг 0: Подготовка модели (query-independent OSCAR)
- Загружаем готовый OSCAR compressor из HF:  
  `naver/oscar-llama-1b` или `naver/oscar-n-layers` (любой из коллекции https://huggingface.co/collections/naver/oscar).
- **Важно для query-independent**:
  - Input: **только документ + [MEM_1] … [MEM_l]** (без query!).
  - В коде OSCAR они вставляют MEM-токены в конец промпта. Просто делаем `input_ids = doc_tokens + mem_token_ids`.
  - Forward → берём `last_hidden_state[:, -l:]` (последние l позиций — это MEM).

#### Шаг 1: Данные (самое простое и быстрое)
- **MS MARCO Passage** (triples): ~500k query-positive-neg,ative.  
  Скачать готовые:  
  - `sentence-transformers/msmarco-distilbert-base-v3` или  
  - HuggingFace `ms-marco` dataset + hard negatives (уже есть в `sentence-transformers`).
- Батч: 64–128 query + 1 positive doc + 4–7 hard negatives (in-batch negatives тоже работают).

#### Шаг 2: Forward pass в тренировочном цикле
```python
# Для документа (query-independent)
doc_input = tokenizer(doc_text, ...).input_ids
mem_input = [mem_token_id] * l
full_input = doc_input + mem_input
outputs = compressor(input_ids=torch.tensor(full_input).unsqueeze(0).to(device))
mem_hiddens = outputs.last_hidden_state[:, -l:, :]   # [1, l, hidden_dim]

doc_emb = projector(mem_hiddens)   # [1, embed_dim]

# Для query (просто и эффективно)
query_input = tokenizer(query_text, ...).input_ids
q_outputs = compressor(input_ids=query_input_tensor)   # тот же backbone!
query_emb = q_outputs.last_hidden_state.mean(dim=1)    # mean-pool (или CLS если добавим)
# Можно добавить отдельный маленький projector для query, но для первого эксперимента — не нужно
```

#### Шаг 3: Loss — классический contrastive (InfoNCE)
Используем **SentenceTransformers** — самый быстрый способ:
```python
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer("...")  # обёртка
model[0].auto_model = compressor  # frozen
model[1] = MEMProjector(...)      # наш проектор как head

train_loss = losses.MultipleNegativesRankingLoss(model)  # или ContrastiveLoss
```

Или чистый HF Trainer с кастомным loss — тоже 20 строк.

#### Шаг 4: Обучение (параметры для «быстро и просто»)
- Epochs: 1–3 (MS MARCO уже большой).
- LR: 1e-4 или 5e-4 (AdamW).
- Batch size: 64–128 (на A100/H100).
- Gradient accumulation если память жмёт.
- Mixed precision (bf16).
- Время: на 1B-модели + только проектор — **2–6 часов** на одной GPU.

### 3. Как оценивать (тоже быстро)
- MS MARCO dev (MRR@10, Recall@10).
- BEIR (subset: NFCorpus, SciFact, TREC-COVID) — чтобы увидеть, не переобучились ли.
- Сравниваем три бейзлайна:
  1. Чистый query-independent OSCAR MEM (mean без проектора).
  2. Наш проектор.
  3. Классический BGE/E5-small (чтобы понять потолок).

### 4. Ещё более быстрый «proof-of-concept» (30 минут)
Если хочешь запустить за полчаса:
- Возьми маленький OSCAR-N-Layers (N=5–8).
- Возьми только 50k примеров из MS MARCO.
- Обучи **только** проектор (LoRA не нужен — MLP 2–3M параметров).
- Используй `sentence-transformers` trainer.

Хочешь — я прямо сейчас набросаю полный Colab-ready скрипт (с import'ами и всем) под конкретную модель из HF.  

Скажи:
1. Какую именно OSCAR-модель берём (oscar-llama-1b или n-layers)?
2. Embed_dim хочешь 768 (как BGE) или 1024/4096?
3. Mean-pool MEM или concat/last?

И я дам готовый код + команду запуска.  
Готовы запускать первый эксперимент? 🚀

**InfoNCE loss** (Information Noise-Contrastive Estimation) — это **стандартный contrastive loss**, который используется в 99 % современных embedding-моделей для retrieval (BGE, E5, GTE, Snowflake, ColBERT и т.д.).

### 1. Математическая формула

Для одного positive pair (query \( q \), positive документ \( d^+ \)) и \( K \) negative документов \( d^-_k \) loss выглядит так:

\[
\mathcal{L}_{\text{InfoNCE}}(q, d^+, \{d^-_k\}_{k=1}^K) = -\log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\exp(\text{sim}(q, d^+)/\tau) + \sum_{k=1}^K \exp(\text{sim}(q, d^-_k)/\tau)}
\]

где:
- \(\text{sim}(u, v) = \frac{u^\top v}{\|u\| \|v\|}\) — cosine similarity (обычно после LayerNorm),
- \(\tau\) — temperature (гиперпараметр, обычно 0.01–0.1),
- знаменатель — это **softmax** по всем кандидатам (positive + negatives).

В батче из \( N \) пар loss усредняется по всем queries:

\[
\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{\text{InfoNCE}}(q_i, d^+_i, \{d^-_{i,j}\})
\]

### 2. Интуиция (почему именно так)

InfoNCE — это **softmax-классификатор** с «шумом»:
- Модель должна максимизировать сходство positive пары.
- Одновременно минимизировать сходство со всеми negatives в батче (включая in-batch negatives).
- Temperature \(\tau\) контролирует «остроту» распределения: маленький \(\tau\) → модель сильнее штрафует даже слабые negatives.

Это эквивалентно максимизации **lower bound на mutual information** между query и positive документом (отсюда и название Information NCE).

### 3. Как это работает именно в нашем случае (проектор OSCAR)

- Query embedding: `q_emb = compressor(query).mean(dim=1)` (или отдельный projector).
- Doc embedding: `d_emb = projector(mem_hiddens)` — **только это** мы обучаем.
- Positive: `(q_emb, d_emb_positive)`.
- Negatives: in-batch (остальные документы в батче) + опционально hard negatives (BM25 или cross-encoder).

Loss заставляет проектор «вытягивать» MEM-токены именно в то пространство, где cosine(q, d) хорошо ранжирует релевантные документы.

### 4. Полная формула в векторном виде (PyTorch-style)

```python
# q_emb, d_emb_positive, d_emb_negatives — все [batch, embed_dim], normalized
sim_pos = (q_emb * d_emb_positive).sum(dim=-1) / tau          # [batch]
sim_neg = torch.matmul(q_emb, d_emb_negatives.T) / tau        # [batch, batch]  (in-batch negatives)

# logits = [positive + all negatives]
logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)    # [batch, 1 + batch_size-1]

labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # positive всегда в колонке 0
loss = F.cross_entropy(logits, labels)
```

Или ещё проще — через `SentenceTransformers` (как я предлагал раньше):

```python
loss = losses.MultipleNegativesRankingLoss(model, scale=1/tau)
```

### 5. Ключевые гиперпараметры и трюки

| Параметр          | Типичное значение | Что делать если…                          |
|-------------------|-------------------|-------------------------------------------|
| \(\tau\)          | 0.02–0.05         | слишком большой → embeddings «размазываются» |
| Кол-во negatives  | in-batch 32–256   | больше → лучше качество (но память)       |
| Hard negatives    | 4–7 на query      | сильно ускоряет сходимость                |
| Нормализация      | LayerNorm / L2    | обязательно перед cosine                  |
| Scale (в ST)      | 20–50             | эквивалент 1/τ                            |

### 6. Почему именно InfoNCE, а не Triplet / CosineEmbeddingLoss

- Triplet (margin) намного слабее (требует explicit hard negative mining).
- InfoNCE использует **все** negatives в батче одновременно → эффективнее.
- Легко масштабируется на миллионы пар (MS MARCO).

Если хочешь, могу сразу дать:
- готовый PyTorch-код loss-функции под твой `MEMProjector`,
- или как добавить hard negatives mining в training loop,
- или сравнение с `MultipleNegativesRankingLoss` из SentenceTransformers.

Скажи, на каком уровне углубиться дальше (формула → код → эксперимент).