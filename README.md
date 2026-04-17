# Projected Token

Исследовательский проект по изучению методов сжатия контекста для RAG (Retrieval-Augmented Generation). Магистерская диссертация.

## Методы сжатия контекста

Проект реализует и сравнивает несколько подходов к сжатию контекста:

### 1. OSCAR (`evaluation/models/oscar/`)
Метод из статьи [arxiv:2504.07109](https://arxiv.org/abs/2504.07109). Сжимает документы в латентные представления фиксированного размера с помощью compressor-компонента. LLM генерирует ответы непосредственно из сжатого контекста.

**Ключевые файлы:**
- `oscar.py` - основная модель с методами `compress_documents()` и `generate_from_compressed_documents_and_questions()`

### 2. PISCO (`evaluation/models/pisco/`)
Реализация из [naver-ai/pisco](https://github.com/naver-ai/pisco). Использует архитектуру COCOM с memory tokens для сжатия документов в латентное пространство.

**Ключевые файлы:**
- `pisco.py` - обертка над COCOM моделью с методами `prepare_encoder_inputs()` и `generate()`

### 3. xRAG (`evaluation/models/xrag/`)
Метод xRAG ([xRAG](https://github.com/AlphaLLM/xRAG)) - инжектирует эмбеддинги документов через специальный токен `[XRAG]` в LLM. Использует отдельный retriever (SFR-Embedding-Mistral) для получения эмбеддингов.

**Ключевые файлы:**
- `xrag.py` - основная модель
- `xmistral.py` - модифицированный Mistral LLM с поддержкой xRAG токена
- `sfr.py` - retriever модель

### 4. RAG (`evaluation/models/rag/`)
Стандартный RAG без сжатия - baseline метод. Документы передаются в LLM напрямую.

### 5. SimpleLLM (`evaluation/models/simple_llm.py`)
Простой LLM без retrieval - baseline метод для сравнения.

## Структура проекта

```
last_projected-token/
├── configs/                    # Конфигурации для разных экспериментов
│   ├── oscar_7b.yaml          # OSCAR на Qwen2-7B
│   ├── pisco_mistral.yaml     # PISCO на Mistral
│   ├── xrag.yaml              # xRAG конфиг
│   ├── qa_*.yaml              # QA эксперименты
│   └── rag_qwen*.yaml         # RAG baseline
├── evaluation/
│   ├── cli.py                 # Главный CLI интерфейс (click)
│   ├── models/                # Реализации моделей (LLM для генерации)
│   │   ├── oscar/
│   │   ├── pisco/
│   │   ├── xrag/
│   │   ├── rag/
│   │   └── simple_llm.py
│   ├── encoders/              # Энкодеры (сжатие документов в векторы)
│   │   ├── encoder.py         # Базовый класс Encoder
│   │   └── oscar.py           # OscarEncoder
│   ├── metrics/               # Метрики оценки
│   │   ├── align_score/
│   │   ├── gpt_score/
│   │   └── qa_score/
│   └── datasets/              # Датасеты (PopQA)
├── scripts/                   # Shell скрипты для запуска
│   ├── run_all_experiments.py
│   ├── run_qa_evaluation.sh
│   └── run_qa_generation.sh
├── evaluate_*.py              # Скрипты оценки
└── run_pisco_paraphrase.py    # Запуск PISCO парафразирования
```

## Метрики оценки

### Простые метрики (`evaluate_simple.py`)
- **Jaccard similarity** - пересечение/объединение токенов
- **Character similarity** - посимвольное сходство (SequenceMatcher)
- **Word overlap** - процент перекрытия слов
- **Length ratio** - отношение длин

### GPT Judge (`evaluate_gpt_judge_only.py`)
LLM-based оценка фактической согласованности между оригиналом и парафразом:
- `supported` - все факты поддерживаются
- `partially_supported` - частично поддерживается
- `contradicted` - есть противоречия
- `unknown` - недостаточно информации

### QA Evaluation (`evaluation/cli.py eval-qa`)
Оценка качества сгенерированных ответов с использованием LLM judge.

## Использование CLI

### Парафразирование
```bash
python -m evaluation.cli run-paraphrase \
    --config configs/oscar_7b.yaml \
    --input-path data/popqa/test.jsonl \
    --output-path results/paraphrase.jsonl
```

### QA генерация
```bash
python -m evaluation.cli run-qa \
    --config configs/qa_oscar_7b.yaml \
    --input-path data/popqa/test.jsonl \
    --output-path results/qa.jsonl
```

### QA оценка
```bash
python -m evaluation.cli eval-qa \
    --input-path results/qa.jsonl \
    --output results/qa_metrics.json \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3.5-27B
```

## Запуск через Docker

```bash
docker build -t projected-token .
docker run -it --gpus all \
  -v ~/hf_cache:/workspace/data/hf_cache \
  -v $(pwd):/workspace \
  projected-token --help
```

## Датасеты

Проект использует **PopQA** (Popular Knowledge QA) - датасет с вопросами разной популярности. Позволяет исследовать зависимость качества от частоты факта.

## Ключевые классы

### Model (базовый класс)
```python
class Model(ABC):
    QA_DEFAULT_PROMPT_TEMPLATE: str
    PARAPHRASE_DEFAULT_PROMPT_TEMPLATE: str
    
    def __call__(self, document: str, prompt_template: str, model_args: dict) -> str
    def generate_batch(self, documents: List[str], prompt_template, model_args, questions) -> List[str]
```

### Encoder (базовый класс для энкодеров)
```python
class Encoder(ABC):
    @abstractmethod
    def encode(self, documents: List[str], questions: Optional[List[str]] = None) -> torch.Tensor:
        """Сжать документы в латентные векторы [batch_size, latent_dim]"""
        
    @abstractmethod
    def encode_batch(self, documents: List[str], questions: Optional[List[str]] = None) -> List[torch.Tensor]:
        """Сжать документы в список векторов [latent_dim]"""
        
    @property
    def latent_dim(self) -> int:
        """Размерность латентного пространства"""
```

### Использование Encoder
```python
from evaluation.encoders import OscarEncoder

encoder = OscarEncoder(
    model_name_or_path="naver/oscar-qwen2-7B",
    device="cuda:0"
)

documents = ["Документ 1...", "Документ 2..."]
questions = ["Вопрос 1?", "Вопрос 2?"]

# Получить латентные векторы (батч)
latent_vectors = encoder.encode(documents, questions)  # [2, hidden_dim]

# Получить список векторов
latent_list = encoder.encode_batch(documents, questions)  # [tensor, tensor]
```

### OscarModel
- `compress_documents(documents, questions)` - сжатие документов
- `generate_from_compressed_documents_and_questions(questions, compressed_documents)` - генерация из сжатого контекста

### PiscoModel
- `prepare_encoder_inputs(texts, max_length, q_texts)` - подготовка входа энкодера
- `generate(model_input, max_new_tokens)` - генерация

### XRAGModel
- Использует retriever для получения эмбеддингов
- Инжектирует эмбеддинги через xRAG токен в LLM
- Кеширует токенизацию промпта для производительности

## Зависимости

- transformers
- torch
- click
- pyyaml
- tqdm
- openai + instructor (для GPT judge)
- numpy


poetry run python oscar_qa_experiment.py \                                                              
       --max-rows 300 \                                                                                                                      
       --aggregation mean \                                                                                                                  
       --batch-size 4 \                                                                                                                      
       --max-new-tokens 32

poetry run python oscar_qa_experiment.py \
    --max-rows 300 \
    --aggregation mean \
    --batch-size 4 \
    --max-new-tokens 32