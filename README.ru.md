# PyTorchBench


## Основные возможности

- Полная настройка эксперимента через YAML-конфиг
- Автоматическое сохранение чекпоинтов и метрик
- Поддержка возобновления обучения
- Гибкая система конфигурации компонентов
- Встроенное логирование метрик и визуализация прогресса

### Пример YAML Файла

```yaml
experiment:
  name: fruits
  paths:
    root_path: experiments/fruits_classification
    model_path: model.pth
    best_model_path: best_model.pth
    optimizer_path: optimizer.pth
    best_optimizer_path: best_optimizer.pth
    train_metrics_path: train_metrics.csv
    val_metrics_path: val_metrics.csv
    test_metrics_path: test_metrics.csv
    graph_path: graph.png
    config_snapshot: details.yaml
    logs_path: logs.log

logger:
  level: DEBUG

train:
  enabled: true
  epochs: 50

test:
  enabled: true

dataset:
  obj: src.data.datasets.ImageClassificationDataset
  init_params:
    path: "datasets/fruits"
    transforms:
      obj: torchvision.transforms.v2.Compose
      init_params:
        transforms:
          - obj: torchvision.transforms.v2.RGB
            init_params: { }
          - obj: torchvision.transforms.v2.Resize
            init_params:
              size: [ 224, 224 ]
          - obj: torchvision.transforms.v2.ToImage
            init_params: { }
          - obj: torchvision.transforms.v2.ToDtype
            init_params:
              dtype:
                obj: torch.float32
                init_params: null
              scale: true
          - obj: torch.nn.Sequential
            init_params:
              args:
                - obj: torch.nn.Conv2d
                  init_params:
                    in_channels: 3
                    out_channels: 16
                    kernel_size: 3
                - obj: torch.nn.ReLU
                  init_params:
                    inplace: true
  post_params:
    batch_size: 64

model:
  obj: torchvision.models.resnet101
  init_params:
    weights:
      obj: torchvision.models.ResNet101_Weights.DEFAULT
      init_params: null
  post_params:
    out_features: 12
    freeze_pretrained_weights: true

criterion:
  obj: torch.nn.CrossEntropyLoss
  init_params: { }

optimizer:
  obj: torch.optim.Adam
  init_params: { }

scheduler:
  obj: torch.optim.lr_scheduler.ReduceLROnPlateau
  init_params: { }

metric_tracker:
  obj: torchmetrics.Accuracy
  init_params:
    task: multiclass
    out_features: 12
```

**Фабрика компонентов `ObjectFactory` рекурсивно обходит конфиг и динамически формирует объекты.**

### Конфигурация компонентов

**Каждый компонент системы настраивается через три основных параметра:**

#### 1. `obj` (обязательный)

```yaml
obj: полный.путь.к.объекту
```

**Примеры**:

```yaml
obj: torch.nn.CrossEntropyLoss
obj: torchvision.models.resnet50
obj: src.data.datasets.CustomDataset
```

#### 2. `init_params` (опциональный)

**Примеры**:

```yaml
# С параметрами
init_params:
  lr: 0.001
  weights:
    obj: torchvision.models.ResNet50_Weights.DEFAULT
    init_params: null

# Без параметров
init_params: null
```

| Значение     | Поведение                        |
|--------------|----------------------------------|
| Присутствует | Передаются в конструктор объекта |
| null         | Объект не инициализируется       |
| Отсутствует  | Эквивалентно null                |

#### 3. `post_params` (настраивается через pydantic модель)

**Примеры**:

```yaml
post_params:
  batch_size: 64          # Для DataLoader
  num_workers: 4          # Для параллельной загрузки
```
#### Пример дерева файлов

```markdown
experiments/
└── fruits_classification/
├── models/
│ ├── model.pth
│ └── best_model.pth
├── optim/
│ ├── optimizer.pth
│ └── best_optimizer.pth
├── metrics/
│ ├── train_metrics.csv
│ ├── val_metrics.csv
│ └── test_metrics.csv
├── progress.png
├── config_snapshot.yaml
├── logs.log
```

## Установка

### 1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/eugeneliukindev/PyTorchBench.git
   ```

### 2. Установка Poetry (если ещё не установлен)

#### Для Linux/macOS:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Для Windows (PowerShell):

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### 3. Установка зависимостей проекта

```bash
# Переходим в директорию проекта
cd PyTorchBench

# Устанавливаем зависимости через Poetry
poetry install --no-root
```

### 4. (Опционально) Активация виртуального окружения

```bash
poetry shell
```

### 5. Проверка установки

```bash
poetry run python -c "import torch; print(torch.__version__)"
```

## Запуск

```bash
python main.py --config config.yaml
```

## Лицензия

[MIT License](https://mit-license.org/)