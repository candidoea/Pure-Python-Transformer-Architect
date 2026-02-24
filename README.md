# 🏗️ Transformer From Scratch (Python Puro)

Este repositório contém uma implementação educacional completa da arquitetura **Transformer**, construída exclusivamente em **Python puro** (sem o uso de frameworks como PyTorch ou TensorFlow). O objetivo é desvendar a "caixa-preta" dos modelos de linguagem modernos (LLMs) através da implementação direta das equações do artigo original *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*.

## 🎯 Objetivo do Projeto

Demonstrar a mecânica interna de um Transformer, desde operações de álgebra linear básica até o fluxo autoregressivo de geração de texto, mantendo a estabilidade numérica e a fidelidade matemática.

## 🛠️ Arquitetura Implementada

O modelo segue a estrutura clássica de um **Decoder-only** (estilo GPT), apresentando os seguintes componentes:

### 1. Positional Encoding (PE)

Como o Transformer processa tokens em paralelo, ele não possui noção intrínseca de ordem. Utilizamos funções senoidais e cossenoidais para injetar informações de posição:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

### 2. Scaled Dot-Product Attention

O coração do modelo, onde as matrizes de **Query (Q)**, **Key (K)** e **Value (V)** interagem para determinar a relevância contextual:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

* **Causal Masking:** Implementação de uma máscara triangular superior ($-\infty$) para garantir que o modelo não "olhe para o futuro" durante a geração.

### 3. Layer Normalization & Resíduos

Para garantir a estabilidade do treinamento e evitar a degradação do gradiente, aplicamos:

* **Conexões Residuais:** $X_{out} = X + \text{Sublayer}(X)$.
* **Layer Norm:** Normalização por token para manter média $0$ e variância $1$.

### 4. Feed-Forward Network (FFN)

Uma camada densa aplicada individualmente a cada posição, introduzindo não-linearidade através da ativação **ReLU**:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

---

## 📊 Visualização e Diagnóstico

O projeto inclui ferramentas de visualização para validar o comportamento do modelo:

* **Matriz de Atenção 3D:** Visualização por "fibras" de luz que demonstram o fluxo de atenção causal entre tokens.
* **Análise de Loss:** Cálculo de erro via Cross-Entropy (ou MSE simplificado) para medir a precisão da predição em relação a um alvo real. No teste final, obtivemos uma **Loss de 0.7029**, indicando um sistema pronto para o processo de otimização.

---

## 🚀 Como Executar

O projeto está contido em um Jupyter Notebook (`.ipynb`). Basta abrir o arquivo e executar as células sequencialmente. Não existem dependências externas além de bibliotecas padrão do Python (como `math` e `random`) e `matplotlib/plotly` para as visualizações.

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/Pure-Python-Transformer-Architect.git

# Abra o notebook
jupyter notebook TRANSFORMER(PYTHON_PURO).ipynb

```

## 🧠 Conclusão Acadêmica

Este projeto prova que, por trás da complexidade de modelos como o GPT-4, existe uma estrutura elegante de matrizes operando em harmonia. A estabilidade numérica alcançada (média de ativação $0.0000$ após LayerNorm) confirma que a implementação está pronta para receber o algoritmo de *Backpropagation* e evoluir para um treinamento real.

---

**Desenvolvido para fins de estudo profundo em IA e Arquiteturas Neurais.**
