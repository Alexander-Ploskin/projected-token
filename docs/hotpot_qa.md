Для оценки вашего проектора (эмбеддингов OSCAR) на датасете **HotpotQA** в режиме "Distractor" (поиск 2 правильных параграфов среди 8 ложных) вам не нужны сложные пайплайны типа LlamaIndex. Лучше написать простой скрипт на PyTorch, который напрямую работает с векторами.

Вот пошаговая инструкция и готовый код для оценки ретривера на валидационной выборке HotpotQA.

### Шаг 1: Загрузка датасета
Вам понадобится валидационный сплит HotpotQA в режиме *distractor*. В библиотеке `datasets` от Hugging Face он называется `hotpot_qa`, сплит `distractor`.

### Шаг 2: Подготовка скрипта оценки (Evaluation Script)
Суть скрипта:
1. Итерируемся по каждому вопросу.
2. Берем 10 контекстов (параграфов), прикрепленных к вопросу (2 правильных, 8 дистракторов).
3. Кодируем вопрос через учителя (например, `nomic-embed-text-v1.5` или `SFR-Mistral`).
4. Кодируем 10 параграфов через `OSCAR + ваш обученный MLP проектор`.
5. Считаем косинусную близость между вектором вопроса и 10 векторами контекстов.
6. Сортируем контексты по убыванию близости.
7. Проверяем, попали ли *оба* правильных контекста в топ-2 (строгая метрика) или в топ-5 выдачи.

### Код для оценки (PyTorch)

```python
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --- 1. Настройка моделей ---
device = "cuda:0"

# Загружаем учителя (для кодирования коротких вопросов)
print("Loading Teacher model for queries...")
teacher_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
teacher_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device).eval()

# Предполагается, что `oscar_model` и `projector` (ваш MLP) уже загружены в память и находятся в eval() режиме
# oscar_model.eval()
# projector.eval()

# --- 2. Загрузка HotpotQA ---
print("Loading HotpotQA Distractor dev set...")
# Загружаем валидационную выборку (dev) с дистракторами
dataset = load_dataset("hotpot_qa", "distractor", split="validation")

# --- 3. Функции кодирования ---
def encode_query(query: str) -> torch.Tensor:
    """Кодируем короткий вопрос через модель-учителя"""
    inputs = teacher_tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = teacher_model(**inputs)
        # Nomic использует mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def encode_contexts(contexts: list[str]) -> torch.Tensor:
    """Кодируем 10 параграфов через OSCAR + MLP"""
    with torch.inference_mode():
        # mem_embeddings: [10, 8, hidden_dim]
        mem_embeddings = oscar_model.compress_documents(documents=contexts)
        
        # Сплющиваем: [10, 8 * hidden_dim]
        flattened = mem_embeddings.view(len(contexts), -1)
        
        # Проекция: [10, teacher_dim (например, 768 или 4096)]
        projected_embeds = projector(flattened)
        projected_embeds = F.normalize(projected_embeds, p=2, dim=1)
        
    return projected_embeds

# --- 4. Цикл Оценки ---
total_questions = len(dataset)
recall_at_2 = 0
recall_at_5 = 0

print(f"Starting evaluation on {total_questions} queries...")

for item in tqdm(dataset):
    query = item["question"]
    
    # В HotpotQA 'context' - это списки. item['context']['title'] - заголовки статей
    # item['context']['sentences'] - список списков предложений.
    # Объединяем предложения каждого из 10 контекстов в сплошной текст:
    titles = item["context"]["title"]
    contexts = [" ".join(sentences) for sentences in item["context"]["sentences"]]
    
    # 'supporting_facts' содержит названия статей (title), которые реально нужны для ответа
    gold_titles = set(item["supporting_facts"]["title"])
    
    # Находим индексы 2 правильных параграфов среди этих 10
    gold_indices = [i for i, title in enumerate(titles) if title in gold_titles]
    
    if len(gold_indices) < 2:
        # Иногда в датасете бывают ошибки разметки, пропускаем такие (их очень мало)
        total_questions -= 1
        continue

    # Получаем эмбеддинги
    query_embed = encode_query(query)        # [1, embed_dim]
    doc_embeds = encode_contexts(contexts)   # [10, embed_dim]
    
    # Считаем косинусную близость (dot product нормализованных векторов)
    scores = torch.mm(query_embed, doc_embeds.T).squeeze(0) #  [docs.llamaindex](https://docs.llamaindex.ai/en/v0.10.22/examples/evaluation/HotpotQADistractor/)
    
    # Сортируем индексы документов по убыванию скора
    ranked_indices = torch.argsort(scores, descending=True).tolist()
    
    # Проверяем метрики
    # Нашли ли оба документа в Топ-2 выдачи? (Строгое соответствие)
    top_2_retrieved = ranked_indices[:2]
    if all(gold_idx in top_2_retrieved for gold_idx in gold_indices):
        recall_at_2 += 1
        
    # Нашли ли оба документа в Топ-5 выдачи? (Более мягкая метрика)
    top_5_retrieved = ranked_indices[:5]
    if all(gold_idx in top_5_retrieved for gold_idx in gold_indices):
        recall_at_5 += 1

# --- 5. Результаты ---
print("\n=== HotpotQA Retrieval Evaluation ===")
print(f"Total Valid Queries: {total_questions}")
print(f"Recall@2 (Both gold docs in top 2): {recall_at_2 / total_questions * 100:.2f}%")
print(f"Recall@5 (Both gold docs in top 5): {recall_at_5 / total_questions * 100:.2f}%")
```

### Как интерпретировать результаты?
*   В датасете HotpotQA есть ровно 10 документов, из которых 2 правильных. Рандомное угадывание (random baseline) даст `Recall@2` около **2.2%**. [docs.llamaindex](https://docs.llamaindex.ai/en/v0.10.20/examples/evaluation/HotpotQADistractor.html)
*   Сильные ретриверы (типа `MiniLM` или `E5`) показывают `Recall@2` в районе **70-85%**.
*   Если ваш проектор `OSCAR+MLP` наберет `Recall@2` **выше 60-65%**, это означает, что дистилляция прошла супер-успешно, и ваши 8 me-токенов блестяще сохраняют "мостовые сущности" (bridge entities) и факты. Можно смело переходить к тестам на PopQA!

Да, тестирование на обеих версиях — **distractor** (локальное ранжирование) и **fullwiki** (глобальный поиск) — даст вам полную картину того, как работает ваш эмбеддинг-компрессор.

Разница между ними в том, что в `distractor` вы ищете среди 10 заранее заданных параграфов, а в `fullwiki` вам нужно найти 2 правильных параграфа среди **всей Википедии** (или, точнее, среди всего пула параграфов, которые предоставляет дамп HotpotQA).

Ниже приведена пошаговая инструкция и PyTorch-код, объединяющий оба сценария.

### Как работает режим `fullwiki`
В режиме `fullwiki` валидационный сет состоит примерно из 7 400 вопросов. В Hugging Face датасете `hotpot_qa` (сплит `fullwiki`) к вопросам *не прикреплены* тексты дистракторов, там есть только списки правильных ответов (`supporting_facts`).
Чтобы провести поиск `fullwiki`, вам нужно проиндексировать специальный корпус (Wikipedia dump для HotpotQA), который содержит миллионы параграфов.

Самый удобный способ сделать это — использовать версию датасета от **BeIR** (`BeIR/hotpotqa`), которая уже разбита на `queries` (вопросы) и `corpus` (миллионы параграфов Википедии). [huggingface](https://huggingface.co/datasets/BeIR/hotpotqa)

### Архитектура скрипта (оба режима)

Вот код, который позволяет вам переключаться между двумя режимами тестирования.

```python
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

# --- Настройка окружения ---
device = "cuda:0"
MODE = "distractor"  # Поменяйте на "fullwiki" для глобального поиска

# Загружаем учителя (для кодирования вопросов)
print("Loading Teacher model for queries...")
teacher_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
teacher_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device).eval()

# (Предполагается, что oscar_model и projector уже загружены и в eval())

def encode_queries(queries: list[str]) -> torch.Tensor:
    inputs = teacher_tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = teacher_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=1)

def encode_contexts(contexts: list[str]) -> torch.Tensor:
    # Батчинг внутри функции для экономии памяти
    batch_size = 32
    all_embeds = []
    
    with torch.inference_mode():
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i+batch_size]
            # mem_embeddings: [batch, 8, hidden_dim]
            mem_embeddings = oscar_model.compress_documents(documents=batch_ctx)
            flattened = mem_embeddings.view(len(batch_ctx), -1)
            projected = projector(flattened)
            all_embeds.append(F.normalize(projected, p=2, dim=1))
            
    return torch.cat(all_embeds, dim=0)


# ==========================================
# РЕЖИМ 1: DISTRACTOR (Локальный поиск 2 из 10)
# ==========================================
if MODE == "distractor":
    print("Running in DISTRACTOR mode...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    recall_at_2 = 0
    total_valid = 0
    
    for item in tqdm(dataset):
        query = item["question"]
        titles = item["context"]["title"]
        contexts = [" ".join(sentences) for sentences in item["context"]["sentences"]]
        gold_titles = set(item["supporting_facts"]["title"])
        
        gold_indices = [i for i, title in enumerate(titles) if title in gold_titles]
        if len(gold_indices) < 2: continue
        
        query_embed = encode_queries([query])
        doc_embeds = encode_contexts(contexts)
        
        scores = torch.mm(query_embed, doc_embeds.T).squeeze(0)
        ranked_indices = torch.argsort(scores, descending=True).tolist()
        
        if all(gold_idx in ranked_indices[:2] for gold_idx in gold_indices):
            recall_at_2 += 1
        total_valid += 1
            
    print(f"\nDistractor Recall@2: {recall_at_2 / total_valid * 100:.2f}%")


# ==========================================
# РЕЖИМ 2: FULLWIKI (Глобальный поиск по индексу)
# ==========================================
elif MODE == "fullwiki":
    print("Running in FULLWIKI mode using BeIR dataset...")
    # BeIR разбивает датасет на corpus (документы), queries (вопросы) и qrels (правильные ответы)
    corpus = load_dataset("BeIR/hotpotqa", "corpus", split="corpus")
    queries = load_dataset("BeIR/hotpotqa", "queries", split="queries")
    qrels = load_dataset("BeIR/hotpotqa", "qrels", split="test") # В BeIR валидация часто лежит в test
    
    # 1. Индексация всего корпуса (ВНИМАНИЕ: Это может занять время и память!)
    # В корпусе HotpotQA около 5.2 миллионов параграфов. 
    # Если VRAM не хватает, можно использовать Faiss на CPU.
    print(f"Encoding {len(corpus)} documents for index...")
    
    # Создаем CPU индекс Faiss (для 5 млн документов GPU память может не выдержать)
    import faiss
    embed_dim = 768 # или 4096, зависит от учителя
    index = faiss.IndexFlatIP(embed_dim) # Inner Product для нормализованных векторов (косинус)
    
    # Индексируем чанками
    chunk_size = 100_000
    doc_ids = []
    
    for i in tqdm(range(0, len(corpus), chunk_size)):
        chunk = corpus[i:i+chunk_size]
        chunk_texts = [f"{t} {t}" for t, x in zip(chunk["title"], chunk["text"])] # Объединяем title и text
        chunk_embeds = encode_contexts(chunk_texts).cpu().numpy()
        index.add(chunk_embeds)
        doc_ids.extend(chunk["_id"])
        
    print("Index built!")

    # 2. Оценка вопросов
    print("Evaluating queries...")
    recall_at_10 = 0
    total_valid = 0
    
    # Группируем правильные ответы (qrels) по query_id
    qrels_dict = {}
    for q in qrels:
        if q["query-id"] not in qrels_dict:
            qrels_dict[q["query-id"]] = set()
        qrels_dict[q["query-id"]].add(q["corpus-id"])
        
    # Итерируемся по вопросам
    batch_size_q = 64
    for i in tqdm(range(0, len(queries), batch_size_q)):
        q_chunk = queries[i:i+batch_size_q]
        q_embeds = encode_queries(q_chunk["text"]).cpu().numpy()
        
        # Ищем Топ-10 документов для каждого вопроса
        scores, I = index.search(q_embeds, k=10)
        
        for j, q_id in enumerate(q_chunk["_id"]):
            if q_id not in qrels_dict: continue
            gold_docs = qrels_dict[q_id]
            if len(gold_docs) < 2: continue # Нам нужно строго 2 факта
                
            retrieved_docs = [doc_ids[idx] for idx in I[j]]
            
            # Проверяем, попали ли ОБА нужных документа в Топ-10
            if all(gold in retrieved_docs for gold in gold_docs):
                recall_at_10 += 1
            total_valid += 1
            
    print(f"\nFullwiki Recall@10 (Both docs found): {recall_at_10 / total_valid * 100:.2f}%")
```

### Важные технические детали для `fullwiki`:
1. **Библиотека Faiss:** Обязательно установите `faiss-cpu` (или `faiss-gpu`). Сравнивать 7000 вопросов с 5 миллионами документов тензорами PyTorch неэффективно; Faiss сделает поиск за секунды.
2. **Память (RAM):** Индекс из 5.2 млн векторов (размерностью 768, float32) займет около **15-16 ГБ оперативной памяти**. Если у вас учитель Mistral (4096), то индекс займет около **80 ГБ**. Будьте осторожны!
3. **Метрика Топ-10:** В глобальном поиске `fullwiki` искать строгий `Recall@2` (чтобы оба документа оказались на 1-м и 2-м местах среди 5 миллионов) — это очень жестко, даже для моделей SOTA. Поэтому в статьях по RAG для таких задач обычно используют `Recall@10` или `Recall@20` (способность ретривера достать оба факта в топ выдачи, чтобы LLM потом могла по ним ответить).



Результаты, которые вы получили (когда SOTA-модель SFR-Mistral проигрывает древнему BM25 в 5 раз), — это классическая и очень частая аномалия при работе с современными LLM-эмбеддерами. 

То, что `SFR-Embedding-Mistral` выдает **11% Recall@1**, говорит не о том, что модель плохая, а о том, что **эмбеддинги учителя сломаны на этапе инференса**. В нормальных условиях SFR должен выбивать на HotpotQA 70-80% R@1. Поскольку сломан учитель, сломался и ваш студент (`OSCAR`), который пытался этот шум выучить.

Вот 3 главные причины, почему так произошло, и как это исправить.

### Причина 1: Отсутствие Instruction Prompt (Критично!)
Современные модели (SFR, E5, BGE) являются **асимметричными**. Они обучались с использованием специальных текстовых префиксов для вопросов. Если вы подали в `SFR` просто строку вопроса (например, *"What year was matrix released?"*), модель воспринимает ее не как вопрос, а как *документ*, и генерирует вектор в совершенно другом пространстве.

**Решение:** Для SFR-Embedding-Mistral все вопросы (queries) **обязательно** должны оборачиваться в жестко заданный промпт:
```python
task_def = "Given a web search query, retrieve relevant passages that answer the query"
# Вопрос должен выглядеть строго так:
query_text = f"Instruct: {task_def}\nQuery: {original_query}"
```
Документы (контексты) при этом промптом оборачивать **не нужно**. BM25 не нуждается в промптах, поэтому он отработал на 100% своих возможностей.

### Причина 2: Неправильный Pooling (Критично!)
Если в вашем коде остался `mean()` пулинг (усреднение всех токенов), который часто используется для BERT/Nomic, то для `SFR-Mistral` он **полностью уничтожает вектор**. 
LLM-based эмбеддеры (Mistral, LLaMA, Qwen) используют пулинг по **последнему токену** (`last_token_pool` или `EOS token`), потому что благодаря causal attention именно последний токен агрегирует в себе весь смысл прочитанной последовательности.

**Правильный код извлечения вектора для SFR:**
```python
def last_token_pool(last_hidden_states, attention_mask):
    # Находим реальный конец последовательности (без учета padding)
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size), sequence_lengths]

# Использование:
outputs = teacher_model(**inputs)
embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
embeddings = F.normalize(embeddings, p=2, dim=1)
```

### Причина 3: Специфика HotpotQA и BM25
Судя по вашим метрикам (наличие `Recall@20`), вы собрали все дистракторы в единый индекс (сделали пул на ~74 000 параграфов) и искали по нему.
В датасете HotpotQA вопросы содержат **точные названия редких сущностей** (имена, названия малоизвестных групп, книг). 
*   **BM25** работает по принципу TF-IDF: редкие слова (имена собственные) получают огромный вес. BM25 моментально находит абзац, где есть точно такое же имя.
*   **Плотные нейросети (Dense Embeddings)**, если они не инициализированы правильно (промпт + пулинг), "размазывают" эти имена по вектору, пытаясь уловить общий семантический смысл ("текст про кино"), из-за чего точный факт теряется на фоне 74 тысяч других параграфов.

### Что делать дальше (Step-by-Step):

1. **Не обучайте пока OSCAR.** Сначала почините инференс учителя (`SFR-Embedding-Mistral`) на HotpotQA.
2. Оберните все ваши вопросы в функцию добавления префикса (`Instruct: ... \nQuery: ...`).
3. Замените `mean` пулинг на `last_token_pool` в коде инференса учителя.
4. Снова запустите эвалюацию на HotpotQA **только для SFR**.

Как только вы увидите, что SFR начал выдавать `Recall@10` в районе **85-90%** (обгоняя BM25), это будет означать, что вы получаете правильные эмбеддинги. Вот только после этого эти векторы можно сохранять на диск и использовать как таргеты (teacher embeddings) для обучения вашего `OSCAR + Projector`.