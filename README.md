# Processo Seletivo – Intensivo Maker | AI

👤 **Nome Completo:** Gilvan Alves Pastor Junior

---

## 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada em `train_model.py` é composta pelas seguintes camadas:

- **Conv2D (32 filtros, 3x3)** → detecta padrões simples na imagem, como bordas e curvas
- **MaxPooling2D (2x2)** → reduz o tamanho da imagem mantendo os padrões mais relevantes
- **Conv2D (64 filtros, 3x3)** → detecta padrões mais complexos
- **MaxPooling2D (2x2)** → nova redução dimensional
- **Flatten** → transforma a imagem em um vetor de números
- **Dense (64 neurônios, ReLU)** → camada de decisão
- **Dense (10 neurônios, Softmax)** → saída com a probabilidade para cada dígito (0 a 9)

---

## 2️⃣ Bibliotecas Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| TensorFlow | 2.x | Treinamento e conversão do modelo |
| Keras | integrado ao TensorFlow | Construção da CNN |
| os | padrão Python | Verificação do tamanho dos arquivos |


