Отличный выбор! Дистилляция (Knowledge Distillation) значительно упрощает задачу, так как вам больше не нужно собирать hard negatives и использовать сложные contrastive losses. Ваш проектор будет учиться переводить сжатое представление OSCAR в уже "умное" и богатое пространство эмбеддингов учителя (`Salesforce/SFR-Embedding-Mistral`).

Вот пошаговая инструкция по адаптации вашего кода под дистилляцию.

### Шаг 1: Подготовка датасета (только документы)

Для дистилляции вам не нужны вопросы или разметка — только сырые длинные тексты. Идеально подойдет корпус MS MARCO Document.

```python
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class DistillationDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        print("Loading MS MARCO Document Corpus...")
        # Загружаем только сами тексты документов
        self.dataset = load_dataset("Tevatron/msmarco-doc-corpus", split=split)
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Возвращаем просто строку текста
        return self.dataset[idx]["text"]

def collate_fn_distil(batch):
    # Batch - это просто список строк (текстов)
    return {"documents": batch}
```

### Шаг 2: Модификация архитектуры проектора

Поскольку OSCAR выдает 8 me-токенов, а учитель выдает 1 вектор (например, размерностью 4096 для Mistral), проектор должен "сплющивать" эти 8 токенов перед подачей в MLP, чтобы не терять контекст и порядок. Вы также должны убедиться, что выходной размер `embed_dim` совпадает с размерностью учителя.

Внесите следующие изменения в `MEMProjector` или в процесс форварда:

```python
# В вашем MLPTrainer.encode_documents:
def encode_documents_student(self, texts: list[str]) -> torch.Tensor:
    with torch.inference_mode():
        # mem_embeddings: [batch, 8, hidden_dim]
        mem_embeddings = self.oscar_model.compress_documents(documents=texts)
    
    # Сплющиваем токены: [batch, 8 * hidden_dim]
    batch_size = mem_embeddings.size(0)
    flattened = mem_embeddings.view(batch_size, -1)
    
    # MLP должен принимать (8 * hidden_dim) и выдавать (teacher_embed_dim)
    # projector: nn.Linear(8 * hidden_dim, teacher_embed_dim)
    embeddings = self.projector(flattened)
    return embeddings
```

### Шаг 3: Функция потерь (Loss Function)

Для дистилляции эмбеддингов чаще всего используют Mean Squared Error (MSE Loss) вместе с косинусной потерей. Проще всего начать с чистого MSE поверх нормализованных векторов.

```python
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def forward(self, student_embeds: torch.Tensor, teacher_embeds: torch.Tensor) -> torch.Tensor:
        # Обязательная L2 нормализация обоих эмбеддингов [web:69, web:71]
        student_embeds = F.normalize(student_embeds, p=2, dim=-1)
        teacher_embeds = F.normalize(teacher_embeds, p=2, dim=-1)
        
        # Считаем MSE между нормализованными векторами
        loss = F.mse_loss(student_embeds, teacher_embeds)
        return loss
```

### Шаг 4: Интеграция учителя (Teacher Model)

Загружать `SFR-Embedding-Mistral` и `OSCAR` на одной видеокарте (GPU) может не хватить памяти. У вас есть два пути:
1. **Онлайн дистилляция:** Если у вас несколько GPU или много VRAM.
2. **Офлайн дистилляция (Рекомендуется):** Предрасчитать векторы учителя заранее и сохранить их на диск (например, в `.pt` или `.hdf5` файл), а при обучении проектора просто загружать их.

Если вы делаете всё "на лету" (онлайн):

```python
# В __init__ вашего MLPTrainer добавьте:
from transformers import AutoModel
print("Loading Teacher model (SFR-Embedding-Mistral)...")
# Модель учителя всегда в eval mode и заморожена
self.teacher_model = AutoModel.from_pretrained(
    "Salesforce/SFR-Embedding-Mistral", 
    torch_dtype=torch.bfloat16
).to(device).eval()

def encode_documents_teacher(self, texts: list[str]) -> torch.Tensor:
    # Пример вызова для SFR-Mistral (зависит от их токенизатора)
    inputs = self.teacher_tokenizer(texts, max_length=4096, padding=True, truncation=True, return_tensors="pt").to(self.device)
    with torch.inference_mode():
        outputs = self.teacher_model(**inputs)
        # Обычно берется last_hidden_state (например, mean pooling или eos token)
        embeddings = outputs.last_hidden_state.mean(dim=1) 
    return embeddings
```

### Шаг 5: Основной цикл обучения (Train Epoch)

Перепишите `train_epoch` так, чтобы он не требовал негативов и вопросов:

```python
def train_epoch(self, epoch: int) -> dict:
    self.model.train()
    total_loss = 0.0

    pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        documents = batch["documents"] # Только длинные тексты

        # 1. Получаем таргеты от учителя
        with torch.inference_mode():
            teacher_embeds = self.encode_documents_teacher(documents)
            # Отсоединяем граф (хотя inference_mode и так это делает)
            teacher_embeds = teacher_embeds.detach()

        # 2. Получаем предсказания студента (OSCAR -> MLP)
        student_embeds = self.encode_documents_student(documents)

        # 3. Считаем Loss и делаем шаг оптимизатора
        self.optimizer.zero_grad()
        loss = self.loss_fn(student_embeds, teacher_embeds)
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        
        # Логирование
        pbar.set_postfix({"mse_loss": f"{loss.item():.6f}"})
        self.writer.add_scalar("train/mse_loss", loss.item(), self.global_step)
        self.global_step += 1

    return {"loss": total_loss / len(self.train_loader)}
```

### Как применять это на инференсе (PopQA)

Поскольку проектор обучался без вопросов, на инференсе для задачи поиска:
1. Вы кодируете все статьи Википедии (`s_wiki_url`) через `OSCAR + MLP`.
2. Короткие запросы (вопросы PopQA) вы пропускаете напрямую через базовую модель учителя (`SFR-Embedding-Mistral`), **без** OSCAR и **без** MLP. 
Учитель и студент теперь живут в одном семантическом пространстве эмбеддингов, поэтому косинусная близость между запросом учителя и сжатым документом студента будет высокой!

Отличный план. Офлайн-дистилляция — это действительно самый эффективный путь: вы один раз считаете дорогие векторы учителя, сохраняете их на диск, а потом очень быстро итерируетесь по архитектуре и гиперпараметрам вашего MLP-проектора, обучаясь из памяти.

Вот подробная инструкция по реализации офлайн-дистилляции.

### 1. Выбор и размер датасета

Для дистилляции эмбеддингов качество и разнообразие данных важнее их количества. В статьях по дистилляции репрезентаций отмечается, что **50 000 – 100 000** разнообразных документов вполне достаточно для начала обучения компактного проектора. [blog.premai](https://blog.premai.io/data-distillation-10x-smaller-models-10x-faster-inference/)
138 миллионов документов из полного корпуса `msmarco-v2` — это колоссальный избыток, который будет предрасчитываться учителем неделями.

**Рекомендация:** Используйте датасет `Tevatron/msmarco-doc-corpus` с Hugging Face. Скачайте его и сделайте случайную подвыборку (Subset) на **100 000 документов**.

### 2. Шаг 1: Офлайн-генерация таргетов (Teacher Embeddings)

Сначала создадим отдельный скрипт, который прогонит 100k документов через `Salesforce/SFR-Embedding-Mistral` и сохранит результаты. Удобнее всего сохранять векторы в формате HDF5 (библиотека `h5py`), так как он позволяет загружать данные кусками (chunking), не забивая оперативную память.

```python
import torch
import h5py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch.nn.functional as F

def generate_teacher_embeddings():
    device = "cuda:0"
    batch_size = 16 # Настройте под вашу VRAM (Mistral 7B требует много памяти)
    num_samples = 100_000

    print("Loading Dataset...")
    dataset = load_dataset("Tevatron/msmarco-doc-corpus", split="train")
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

    print("Loading Teacher Model...")
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral', torch_dtype=torch.bfloat16).to(device)
    model.eval()

    # SFR-Mistral требует специфического пулинга (last_token_pool) [web:78]
    def last_token_pool(last_hidden_states, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    print("Generating Embeddings...")
    # Создаем HDF5 файл для хранения таргетов и оригинальных текстов
    with h5py.File('teacher_embeddings_100k.h5', 'w') as f:
        # Размерность SFR-Mistral равна 4096
        emb_dataset = f.create_dataset('embeddings', shape=(num_samples, 4096), dtype='float32')
        # Сохраняем тексты, чтобы потом скармливать их OSCAR'у
        text_dataset = f.create_dataset('texts', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))

        for i in tqdm(range(0, num_samples, batch_size)):
            batch_texts = dataset[i:i+batch_size]["text"]
            
            # Токенизация для учителя (max_length=4096)
            inputs = tokenizer(batch_texts, max_length=4096, padding=True, truncation=True, return_tensors="pt").to(device)
            
            with torch.inference_mode():
                outputs = model(**inputs)
                embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                # Нормализуем таргеты, как того требует SFR [web:78]
                embeddings = F.normalize(embeddings, p=2, dim=1)

            # Сохраняем батч на диск
            emb_dataset[i:i+batch_size] = embeddings.cpu().float().numpy()
            text_dataset[i:i+batch_size] = batch_texts

generate_teacher_embeddings()
```

### 3. Шаг 2: Создание HDF5 Dataset для обучения

Теперь, когда у нас есть файл `teacher_embeddings_100k.h5`, напишем `Dataset`, который будет читать тексты (для подачи в OSCAR) и уже готовые таргеты.

```python
from torch.utils.data import Dataset, DataLoader
import h5py

class H5DistillationDataset(Dataset):
    def __init__(self, h5_path, split="train", val_split=0.05):
        self.h5_path = h5_path
        self.file = h5py.File(h5_path, 'r')
        total_samples = self.file['embeddings'].shape[0]
        
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size
        
        # Разделяем индексы на train и val
        if split == "train":
            self.indices = range(0, train_size)
        else:
            self.indices = range(train_size, total_samples)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Читаем данные с диска "на лету" (lazy loading)
        real_idx = self.indices[idx]
        text = self.file['texts'][real_idx].decode('utf-8')
        target_embed = self.file['embeddings'][real_idx]
        
        return {
            "text": text,
            "target": torch.tensor(target_embed, dtype=torch.float32)
        }

def collate_fn_h5(batch):
    texts = [item["text"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])
    return {"texts": texts, "targets": targets}
```

### 4. Шаг 3: Обучение проектора (Student)

Ваш проектор теперь должен принимать сплющенные 8 токенов от OSCAR и выдавать вектор размерностью 4096.

```python
# Обновленный код для train_epoch внутри вашего Trainer
def train_epoch(self, epoch: int):
    self.model.train()
    total_loss = 0.0
    pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        texts = batch["texts"]
        # Таргеты от SFR-Mistral [batch, 4096]
        teacher_embeds = batch["targets"].to(self.device, dtype=torch.bfloat16)

        # 1. Сжимаем тексты через OSCAR
        with torch.inference_mode():
            # mem_embeddings: [batch, 8, hidden_dim]
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)
        
        # 2. Пропускаем через проектор
        batch_size = mem_embeddings.size(0)
        flattened = mem_embeddings.view(batch_size, -1)
        student_embeds = self.projector(flattened) # [batch, 4096]

        # 3. Считаем Loss
        # Нормализуем выход студента (учитель уже нормализован)
        student_embeds = F.normalize(student_embeds, p=2, dim=-1)
        
        # MSE Loss между нормализованными векторами работает лучше всего для дистилляции
        loss = F.mse_loss(student_embeds, teacher_embeds)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"mse_loss": f"{loss.item():.5f}"})

    return {"loss": total_loss / len(self.train_loader)}
```

### Гиперпараметры для дистилляции:
- **Learning Rate:** Для MSE-дистилляции на MLP обычно используют LR повыше, чем для дообучения полных моделей. Начните с `lr = 1e-3` (или `5e-4`) с косинусным расписанием.
- **Batch Size:** Поскольку вы не ограничены памятью модели учителя (вы читаете готовые векторы), ставьте максимально возможный батч-сайз для OSCAR (например, 128 или 256), чтобы градиенты были стабильнее.
- **Архитектура MLP:** Слой должен быть `nn.Linear(8 * oscar_hidden_dim, 4096)`. Можно добавить один скрытый слой с активацией (например, `GELU`), если линейной проекции будет недостаточно для достижения низкого MSE.

Для вашей задачи (дистилляция 8 me-токенов OSCAR в пространство эмбеддингов SFR-Mistral) **Вариант Б (2 слоя с GELU) однозначно предпочтительнее**. 

Вот почему:

### 1. Нелинейность пространства учителя
Вы пытаетесь спроецировать выходы `OSCAR` в семантическое пространство `SFR-Embedding-Mistral`. Модель учителя — это огромная LLM (7B параметров), её векторное пространство сильно нелинейно и обладает сложной геометрией. Один линейный слой (Вариант А) может осуществлять только простые аффинные преобразования (поворот, масштабирование, сдвиг). Этого почти всегда недостаточно, чтобы "дотянуться" до сложного распределения учителя. Нелинейная активация (`GELU`) позволяет студенту выучить более сложные отображения. [arxiv](https://arxiv.org/html/2405.15311v1)

### 2. "Distilling Bottleneck" (Узкое горлышко дистилляции)
Исследования по дистилляции эмбеддингов показывают, что размерность скрытого слоя в проекторе играет критическую роль. Проецирование напрямую (28672 → 4096) создает резкое сужение (bottleneck). 
В Варианте Б вы сначала сжимаете 28672 до 8192 (промежуточное состояние, где сеть может "собрать" признаки вместе, используя нелинейность), а затем до 4096. В статьях по дистилляции репрезентаций часто отмечают, что расширение или плавное сужение размерности скрытого слоя MLP проектора существенно повышает итоговое качество эмбеддингов студента. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2405.15311)

### Рекомендуемая архитектура (Вариант Б)

```python
import torch.nn as nn

class MEMProjector(nn.Module):
    def __init__(self, input_dim=28672, hidden_dim=8192, output_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Очень рекомендуется добавить LayerNorm!
            nn.GELU(),
            # Опционально: nn.Dropout(0.1) если будет переобучаться на 100k примерах
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: [batch, 28672]
        return self.mlp(x)
```

**Дополнительный совет:** Обязательно добавьте `LayerNorm` между линейными слоями (как показано в коде выше). При работе с большими размерностями (8192) дисперсия активаций может сильно улетать, что делает обучение нестабильным. LayerNorm перед GELU стабилизирует градиенты и позволяет использовать более высокий Learning Rate (например, `1e-3`), что критически важно для быстрой сходимости дистилляции.