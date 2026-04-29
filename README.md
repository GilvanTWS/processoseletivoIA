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

### Treinamento
| Métrica | Valor |
|---|---|
| Acurácia final no teste | 99,01% |
| Loss final no teste | 0,0308 |

### Otimização
| Técnica | Tamanho | Redução |
|---|---|---|
| Modelo original (.h5) | 1468.2 KB | - |
| Dynamic Range Quantization | 128.2 KB | ~91% |
| Float16 Quantization | 243.7 KB | ~83% |

O modelo com Dynamic Range Quantization apresentou maior redução de tamanho, sendo o mais indicado para dispositivos com memória muito limitada. O Float16 oferece um equilíbrio entre tamanho e precisão numérica, sendo mais adequado para dispositivos que suportam operações em ponto flutuante de 16 bits.

---

## 5️⃣ Comentários Adicionais

### Decisões técnicas
- Foram utilizadas apenas 2 camadas Conv2D para manter o modelo simples e compatível com as restrições de Edge AI e do pipeline de CI
- O `batch_size` de 64 foi escolhido para equilibrar velocidade de treinamento e uso de memória
- Foi adicionada uma camada `Dropout(0.3)` entre as camadas Dense para reduzir overfitting, o que melhorou a acurácia no teste de 98,74% para 99,01%
- Foram aplicadas duas técnicas de otimização diferentes para demonstrar o trade-off entre tamanho e precisão:
  - **Dynamic Range Quantization** → maior redução (91%), ideal para dispositivos com memória muito limitada
  - **Float16 Quantization** → redução menor (83%), porém mantém maior precisão numérica

### Aprendizados
- Entendi na prática o fluxo completo de Machine Learning: treinamento → salvamento → conversão → otimização
- Aprendi como técnicas de regularização como Dropout melhoram a generalização do modelo
- Compreendi que diferentes técnicas de quantização oferecem trade-offs distintos entre tamanho e desempenho, e que a escolha depende do dispositivo alvo
