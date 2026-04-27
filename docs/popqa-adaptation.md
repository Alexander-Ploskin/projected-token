Это типичная проблема **domain shift** — модель обучена на одной задаче/распределении, но применяется на принципиально другой. Вот детальный анализ и план действий.

## Корень проблемы

Модель `msmarco-distilbert-base-v3` обучена на **MS MARCO** — парах *query → passage*, где запросы — это фразы вроде поисковых запросов Bing, а документы — веб-страницы. PopQA — это совсем другой сигнал: **factual question → Wikipedia title/article**. Распределения принципиально разные: [zilliz](https://zilliz.com/ai-faq/why-is-my-semantic-search-using-sentence-transformer-embeddings-returning-irrelevant-or-bad-results-and-how-can-i-improve-the-retrieval-quality)

| Аспект | MS MARCO | PopQA |
|---|---|---|
| Тип запроса | Информационный поиск (длинный) | Фактический вопрос (короткий) |
| Документ | Веб-пассаж | Wikipedia статья/заголовок |
| Сигнал близости | BM25-based relevance | Entity matching |
| Лексика | Разнородная | Энциклопедическая |

Именно здесь ломается проектор — он выучил «MS MARCO-style proximity», а не «entity-factual proximity». [milvus](https://milvus.io/ai-quick-reference/why-is-my-semantic-search-using-sentence-transformer-embeddings-returning-irrelevant-or-bad-results-and-how-can-i-improve-the-retrieval-quality)

***

## Возможные причины плохого результата

**1. Несоответствие форматов запрос → документ**
Если ты матчишь вопрос ("Who is the president of France?") к заголовку страницы Wikipedia ("Emmanuel Macron"), то это принципиально другая задача, чем query-passage в MSMARCO.

**2. Проектор обучен только на MSMARCO-сигнале**
Если у тебя есть линейный или MLP-проектор поверх эмбеддингов, он переобучился на стиле MSMARCO-запросов и не обобщается. [zilliz](https://zilliz.com/ai-faq/why-is-my-semantic-search-using-sentence-transformer-embeddings-returning-irrelevant-or-bad-results-and-how-can-i-improve-the-retrieval-quality)

**3. Нормализация эмбеддингов**
Если эмбеддинги не нормализованы перед cosine similarity — результаты могут быть искажены. [milvus](https://milvus.io/ai-quick-reference/why-is-my-semantic-search-using-sentence-transformer-embeddings-returning-irrelevant-or-bad-results-and-how-can-i-improve-the-retrieval-quality)

***

## Как дообучить проектор на PopQA

### Шаг 1: Подготовка данных для PopQA
```python
from datasets import load_dataset

popqa = load_dataset("akariasai/PopQA")
# Формируем пары: (question, wikipedia_page_title) или (question, wikipedia_passage)
# Для страниц Wikipedia — подгрузи Wikipedia через ir_datasets или DPR wiki dump
```

### Шаг 2: Дообучение с mixed-domain трипплетами

Не выбрасывай MSMARCO-данные — **смешай** их с PopQA-трипплетами, чтобы не забыть предыдущие знания (catastrophic forgetting):

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# Трипплеты: (question, positive_wiki_passage, hard_negative)
train_examples_popqa = [
    InputExample(texts=[q, pos_wiki, hard_neg])
    for q, pos_wiki, hard_neg in popqa_triplets
]

# Смешать с частью MSMARCO (например, 30% MSMARCO + 70% PopQA)
mixed_examples = msmarco_sample + train_examples_popqa

train_dataloader = DataLoader(mixed_examples, shuffle=True, batch_size=64)

model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v3")
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True
)
```

### Шаг 3: Hard negatives — ключевое

Для PopQA важно майнить **hard negatives** — страницы Wikipedia, которые семантически похожи, но фактически неверны. Используй BM25 или сам эмбеддинг-ретривер для их поиска:

```python
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr')
# BM25 top-10 → берёшь все кроме правильного как hard negatives
```

### Шаг 4: Дообучение только проектора (если backbone заморожен)

Если ты дообучаешь только проектор (а base-модель заморожена), то проблема глубже — backbone не выдаёт нужные representations для PopQA. В этом случае нужно либо разморозить несколько последних слоёв backbone, либо использовать адаптеры (LoRA):

```python
# Разморожаем последние 2 трансформер-блока DistilBERT
for name, param in model.named_parameters():
    if "transformer.layer.5" in name or "transformer.layer.4" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

***

## Диагностика перед дообучением

Прежде чем обучать, стоит проверить:

- **Baseline**: какой результат даёт тот же `msmarco-distilbert-base-v3` *без* твоего проектора на PopQA? Если он тоже плохой — проблема в backbone, а не в проекторе
- **BM25 baseline**: насколько хорош простой BM25 на PopQA? Это нижняя планка для сравнения
- **Нормализация**: убедись, что `model.encode(..., normalize_embeddings=True)` [milvus](https://milvus.io/ai-quick-reference/why-is-my-semantic-search-using-sentence-transformer-embeddings-returning-irrelevant-or-bad-results-and-how-can-i-improve-the-retrieval-quality)

Можешь уточнить: твой проектор — это линейный слой поверх эмбеддингов, или ты fine-tune весь backbone? От этого сильно зависит стратегия.

У вас возникла классическая проблема генерализации: проектор (компрессор), обученный на одном типе QA или retrieval задачи (MS MARCO), не "распаковывает" нужную информацию для задач на фактологические знания (PopQA).

В архитектуре **OSCAR (Online Soft Compression And Reranking)** документы сжимаются в `8 mem-токенов` (при факторе сжатия x16 для фрагментов в 128 токенов). При этом обучение такого компрессора требует специфического подхода, поскольку он должен научиться "прокидывать" нужные факты через *bottleneck* mem-токенов к генератору (LLM). [arxiv](https://arxiv.org/html/2504.07109v1)

Вот лучшая стратегия по исправлению ошибки, базирующаяся на архитектуре OSCAR и природе PopQA.

## Почему ломается на PopQA?

MS MARCO — это задача классического поиска (information retrieval). В ней модель учится кодировать общую релевантность текста запросу. PopQA, напротив, — это **entity-centric factual QA** (вопросы о фактах с длинным хвостом распределения из Wikidata). [llm-stats](https://llm-stats.com/benchmarks/popqa)
Когда вы обучали проектор на MS MARCO, mem-токены научились кодировать семантическую близость, но не научились сохранять точные имена сущностей (entities, годы, факты), которые критически важны для генератора при ответе на PopQA. Компрессор просто "выбрасывает" эти детали как шум, потому что на MS MARCO это не требовалось.

## Стратегия исправления (Fine-tuning OSCAR-style)

Обучать или дообучать mem-токены (и проектор) для RAG нужно не через стандартный contrastive loss, а через **Knowledge Distillation от учителя (генератора)**. [arxiv](https://arxiv.org/html/2504.07109v1)

### 1. Переход на Loss дистилляции (Ключевое отличие OSCAR)
В оригинальной статье OSCAR компрессор обучается **без ground truth меток**, а через кросс-энтропию (cross-entropy distillation) от LLM-учителя (golden teacher). [arxiv](https://arxiv.org/html/2504.07109v1)
Вместо того чтобы учить проектор предсказывать MS MARCO-релевантность, сделайте следующее:
- Возьмите не сжатый документ и вопрос, подайте их в мощную LLM (например, в не сжатый генератор). Получите логиты ответа.
- Прогоните тот же документ через ваш компрессор (получив 8 mem-токенов).
- Подайте эти 8 токенов + вопрос в замороженный генератор.
- **Функция потерь:** минимизируйте KL-дивергенцию (или Cross-Entropy) между логитами "учителя" (с полным текстом) и "ученика" (с 8 mem-токенами). [gist](https://gist.science/paper/2504.07109)

### 2. Смешанный датасет (Mixed-Domain Training)
Нельзя дообучать *только* на PopQA, иначе модель забудет, как делать общий retrieval (catastrophic forgetting).
Создайте датасет, состоящий из:
- **30-40% MS MARCO** (чтобы сохранить робастность к длинным запросам).
- **60-70% QA датасетов**, сфокусированных на фактах: *PopQA, Natural Questions (NQ), TriviaQA*. [arxiv](https://arxiv.org/html/2504.07109v1)
Важно, чтобы при обучении компрессор видел пары «Вопрос — Документ с Википедии, содержащий точный факт», и учитель заставлял генератор выводить этот факт из 8 токенов.

### 3. Query-Dependent Компрессия
В OSCAR компрессия является *зависимой от запроса (query-dependent)*. Если ваш текущий проектор сжимает документ в 8 токенов *независимо* от вопроса (как обычный эмбеддинг документа), на PopQA это приведет к провалу. Документ содержит много фактов, и 8 токенов не хватит для их хранения. [huggingface](https://huggingface.co/api/resolve-cache/models/naver/oscar-mistral-7B/0491ccd6a06780325d8c4df5eaf585b65bcbf1b4/README.md?download=true&etag=%2276e696ab49af3cea21ea459a2c071e8e215eae53%22)
**Решение:** Убедитесь, что на вход компрессора подается конкатенация `[Query] + [Document] + [8x MEM tokens]`. Тогда проектор научится извлекать из документа ровно ту сущность, о которой спрашивают в PopQA.

### 4. Обучение Rerank-токена (Опционально, но полезно)
Если вы используете RAG, добавьте `[RR]` (Rerank) токен на этапе сжатия. Помимо дистилляции ответа от генератора, добавьте loss-компоненту, где `[RR]` токен предсказывает релевантность (по BM25 или Cross-Encoder). Это заставит первые слои проектора выучить, содержит ли документ вообще нужный факт для PopQA, прежде чем пытаться его сжать. [gist](https://gist.science/paper/2504.07109)

### Резюме пайплайна:
1. Заморозьте веса вашего LLM-генератора.
2. Подготовьте батчи из (Query, Document), где данные собраны из MS MARCO + PopQA + Natural Questions.
3. Прогоните компрессор (Query + Doc -> 8 mem-tokens).
4. Обучайте веса компрессора через Teacher Forcing (Cross Entropy Loss), заставляя генератор поверх 8 токенов выдавать тот же ответ, который выдает не сжатая модель на полном тексте. [gist](https://gist.science/paper/2504.07109)