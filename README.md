# 🖼️ Classificação de Imagens no CIFAR-10 com MLP e CNN

**Centro de Informática (CIn) - Universidade Federal de Pernambuco (UFPE)** 

**Disciplina:** Redes Neurais (IF702)

---

## Equipe
* Antonio Carolino
* Beatriz Mergulhão
* Luiza Trigueiro
* João Ebbers

---

## 📌 Sobre o Projeto

Este projeto explora a construção, o treinamento e a otimização de Redes Neurais para a classificação de imagens do dataset **CIFAR-10** (10 classes de objetos e animais). A pesquisa foi dividida em duas frentes principais: modelos **Multi-Layer Perceptron (MLP)** e **Convolutional Neural Networks (CNN)**.

Utilizamos a biblioteca **PyTorch** e aplicamos extensivamente o framework **Optuna** para a busca automatizada de hiperparâmetros (HPO), visando encontrar as melhores arquiteturas, funções de ativação, otimizadores e técnicas de regularização.

---

## 🛠️ Metodologia e Pipeline de Dados

O pipeline de processamento foi projetado para maximizar a generalização dos modelos e evitar *overfitting*:

* **Data Augmentation:** Aplicação de `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`, `RandomRotation` e `RandomErasing` para aumentar a robustez espacial e de cores dos modelos CNN.
* **Normalização:** Padronização dos tensores de imagem com média e desvio padrão específicos do CIFAR-10.
* **Otimização Automatizada (Optuna):** Uso de amostradores TPE (*Tree-structured Parzen Estimator*) e *MedianPruner* para interromper execuções de baixo desempenho prematuramente (Early Stopping inter-trials).
* **Métricas de Avaliação:** Acurácia, Acurácia Balanceada, Precision, Recall e F1-Score (ponderados), além da geração de Matrizes de Confusão.

---

## 🧠 Frente 1: Redes Multi-Layer Perceptron (MLP)



A exploração com MLPs foi focada em entender como diferentes hiperparâmetros e topologias afetam redes densas ao lidar com imagens achatadas (3072 features de entrada). Realizamos 7 experimentos principais:

1. **Variação do Batch Size:** Avaliação do impacto de diferentes tamanhos de lote (32, 64, 128, 256) na velocidade de convergência e estabilidade.
2. **Profundidade e Dropout:** Testes com redes mais profundas (até 5 camadas ocultas) aplicando *Dropout* forte (0.4 a 0.7) para controlar a variância.
3. **Otimização Completa (Arquitetura Dinâmica):** Busca conjunta pelo número ideal de camadas, neurônios por camada, *Learning Rate* e *Dropout*.
4. **Otimizadores:** Comparação direta de desempenho e convergência entre Adam e SGD com *Momentum*.
5. **Weight Decay (Regularização L2):** Busca pela penalidade L2 ideal para mitigar pesos excessivamente grandes e melhorar a generalização.
6. **Ativações ReLU vs. Tanh:** Comparativo da não-linearidade e dissipação de gradientes.
7. **Ativações ReLU vs. Sigmoid:** Confirmação da superioridade do ReLU frente aos problemas de saturação da Sigmoid.

---

## 👁️ Frente 2: Redes Neurais Convolucionais (CNN)



As CNNs aproveitam a topologia 2D das imagens. Partindo de uma adaptação da arquitetura clássica LeNet-5, escalamos para redes mais profundas e arquiteturas dinâmicas.

* **Busca de Otimizadores e Schedulers:** Avaliação de Adam, RMSprop e SGD (com Nesterov). Implementação de estratégias avançadas de taxa de aprendizado, como *Cosine Annealing* e *Warmup* linear.
* **Tuning de Arquitetura Espacial:** Otimização via Optuna do número de camadas convolucionais (2 a 4) e quantidade de filtros (32 a 128) por camada.
* **Estudo Analítico de Operadores de Convolução:** Avaliação isolada de diferentes blocos convolucionais:
  * Convoluções Padrão (*Standard*)
  * *Depthwise Separable Convolutions*
  * *Grouped Convolutions*
  * *Dilated Convolutions*
  * Convoluções 1x1 (*Pointwise*)
  * Blocos Residuais (*Skip Connections*)
* **Controle de Estabilidade e Overfitting:** Construção do modelo `CNN3` integrando `BatchNorm2d`, `Dropout`, `Label Smoothing` (0.1) e precisão mista (*Automatic Mixed Precision - AMP*) para acelerar o treinamento na GPU e estabilizar os gradientes.

