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

---

## 3️⃣ Técnica de Otimização do Modelo

Foi utilizada a técnica de **Dynamic Range Quantization**, disponível no TensorFlow Lite.

Essa técnica converte os pesos do modelo de ponto flutuante (float32) para inteiros de 8 bits (int8), reduzindo significativamente o tamanho do modelo sem a necessidade de um dataset de calibração.

É a técnica mais indicada para aplicações de **Edge AI**, pois permite rodar modelos em dispositivos com pouca memória e processamento limitado, como microcontroladores e sistemas embarcados.

---

## 4️⃣ Resultados Obtidos

| Métrica | Valor |
|---|---|
| Acurácia final no teste | 98,62% |
| Tamanho do modelo original (.h5) | 1467,1 KB |
| Tamanho do modelo otimizado (.tflite) | 128,2 KB |
| Redução de tamanho | ~91% |

O modelo atingiu alta acurácia em apenas 5 épocas de treinamento, demonstrando que a arquitetura CNN escolhida é eficiente para o problema de classificação de dígitos manuscritos. A otimização reduziu o modelo em aproximadamente 91%, tornando-o adequado para execução em dispositivos Edge.

