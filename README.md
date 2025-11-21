# TrabalhoFinal_VisaoComputacional

Projeto final de Visão Computacional com ambiente completo de preparação técnica usando PyTorch.

## 📋 Descrição

Este repositório contém a infraestrutura completa para desenvolvimento de projetos de visão computacional, incluindo:

- ✅ Ambiente configurado com PyTorch e dependências essenciais
- ✅ Pipeline de pré-processamento padronizado (224x224, normalização)
- ✅ Data loaders com shuffle para treinamento
- ✅ Scripts de verificação de GPU/CPU
- ✅ Exemplos de uso e testes

## 🚀 Início Rápido

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Moura196/TrabalhoFinal_VisaoComputacional.git
cd TrabalhoFinal_VisaoComputacional
```

2. Execute o script de configuração interativa:
```bash
python setup_environment.py
```

Ou instale manualmente as dependências:
```bash
pip install -r requirements.txt
```

### Verificação do Ambiente

Execute o script de verificação para testar as importações e disponibilidade de GPU/CPU:

```bash
python verify_environment.py
```

Este script irá:
- ✓ Verificar todas as importações de pacotes
- ✓ Detectar se GPU está disponível
- ✓ Testar operações básicas de tensores
- ✓ Executar um treinamento "hello world"

## 📦 Dependências

### Deep Learning
- **PyTorch** (>=2.0.0) - Framework principal de deep learning
- **Torchvision** (>=0.15.0) - Utilitários e datasets para visão computacional

### Processamento de Dados
- **NumPy** (>=1.24.0) - Computação numérica
- **Pandas** (>=2.0.0) - Manipulação de dados
- **Pillow** (>=10.0.0) - Processamento de imagens

### Machine Learning
- **Scikit-learn** (>=1.3.0) - Algoritmos de machine learning

### Visualização
- **Matplotlib** (>=3.7.0) - Criação de gráficos
- **Seaborn** (>=0.12.0) - Visualização estatística

### Utilidades
- **tqdm** (>=4.65.0) - Barras de progresso

## 🔧 Pipeline de Pré-processamento

O módulo `data_preprocessing.py` fornece um pipeline padronizado para preparação de imagens:

### Características Principais

- **Resize automático**: Todas as imagens são redimensionadas para 224x224 pixels
- **Normalização por canal**: Utiliza estatísticas do ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Data augmentation**: Opcional para treinamento (flip horizontal, rotação, color jitter)
- **Data loaders**: Com shuffle automático para conjunto de treino

### Uso Básico

```python
from data_preprocessing import get_preprocessing_pipeline, create_data_loaders
from PIL import Image

# Criar preprocessador para treino (com augmentation)
train_preprocessor = get_preprocessing_pipeline(mode='train')

# Criar preprocessador para avaliação (sem augmentation)
eval_preprocessor = get_preprocessing_pipeline(mode='eval')

# Processar uma imagem
image = Image.open('caminho/para/imagem.jpg')
tensor = train_preprocessor(image)  # Shape: (3, 224, 224)

# Criar dataset customizado
from data_preprocessing import CustomImageDataset

dataset = CustomImageDataset(
    images=lista_de_imagens,
    labels=lista_de_labels,
    transform=train_preprocessor
)

# Criar data loaders
loaders = create_data_loaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=32,
    shuffle_train=True  # Shuffle ativo para treino
)

# Iterar sobre batches
for images, labels in loaders['train']:
    # images: (batch_size, 3, 224, 224)
    # labels: (batch_size,)
    pass
```

### Exemplo Completo

```python
import torch
from torch.utils.data import Dataset
from data_preprocessing import (
    get_preprocessing_pipeline,
    CustomImageDataset,
    create_data_loaders
)

# 1. Preparar preprocessamento
train_transform = get_preprocessing_pipeline(mode='train')
val_transform = get_preprocessing_pipeline(mode='eval')

# 2. Criar datasets
train_dataset = CustomImageDataset(
    images=train_images,
    labels=train_labels,
    transform=train_transform
)

val_dataset = CustomImageDataset(
    images=val_images,
    labels=val_labels,
    transform=val_transform
)

# 3. Criar data loaders
loaders = create_data_loaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=32,
    num_workers=4,
    shuffle_train=True  # Shuffle habilitado para treino
)

# 4. Treinar modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(num_epochs):
    for images, labels in loaders['train']:
        images = images.to(device)
        labels = labels.to(device)
        
        # Seu código de treinamento aqui
        outputs = model(images)
        loss = criterion(outputs, labels)
        # ...
```

## 📊 Estrutura do Projeto

```
TrabalhoFinal_VisaoComputacional/
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências do projeto
├── setup_environment.py         # Script de configuração interativa
├── verify_environment.py        # Script de verificação do ambiente
├── data_preprocessing.py        # Pipeline de pré-processamento
└── .gitignore                   # Arquivos ignorados pelo git
```

## 🧪 Testes

### Testar Importações e GPU/CPU

```bash
python verify_environment.py
```

### Testar Pipeline de Pré-processamento

```bash
python data_preprocessing.py
```

## 💡 Características do Pipeline

### 1. Padronização de Entrada

- **Dimensões fixas**: 224x224 pixels (padrão para transfer learning)
- **Formato**: Tensor PyTorch (C, H, W) = (3, 224, 224)
- **Tipo de dados**: Float32
- **Intervalo de valores**: Normalizado usando estatísticas ImageNet

### 2. Normalização por Canal

```python
# Estatísticas do ImageNet
mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]   # RGB
```

### 3. Data Augmentation (Treino)

- Flip horizontal aleatório (p=0.5)
- Rotação aleatória (±15 graus)
- Color jitter (brightness, contrast, saturation, hue)

### 4. Data Loaders

- **Shuffle**: Habilitado automaticamente para treino
- **Pin memory**: Habilitado quando GPU disponível
- **Drop last**: True para treino, False para validação
- **Num workers**: Configurável (padrão: 4)

## 🎯 Próximos Passos

Após configurar o ambiente, você pode:

1. **Preparar seus dados**: Organize suas imagens e labels
2. **Criar dataset customizado**: Use `CustomImageDataset` como template
3. **Definir modelo**: Crie ou carregue um modelo de rede neural
4. **Treinar**: Use os data loaders para treinar seu modelo
5. **Avaliar**: Use o data loader de validação para avaliar o modelo

## 📝 Notas

- O pipeline é otimizado para transfer learning com modelos pré-treinados no ImageNet
- Para outros casos de uso, você pode customizar os parâmetros de normalização
- GPU é opcional mas recomendado para treinamento mais rápido
- Os data loaders usam `pin_memory=True` automaticamente quando GPU está disponível

## 🤝 Contribuindo

Para contribuir com este projeto:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto é parte do trabalho final de Visão Computacional.

## 👥 Autores

- Moura196

## 🙏 Agradecimentos

- PyTorch team pelo excelente framework
- ImageNet dataset pelos dados de treinamento e estatísticas de normalização