# Determinantes Socioeconômicos da Criminalidade nos Municípios do Estado de São Paulo

Este projeto apresenta uma pipeline completa de análise de dados para investigar os determinantes socioeconômicos da criminalidade nos municípios do estado de São Paulo. A análise utiliza técnicas estatísticas, aprendizado de máquina e métodos de análise espacial para compreender padrões associados às taxas de homicídio.

O projeto foi desenvolvido como parte de um estudo acadêmico na área de Data Science e Analytics.

---

# Objetivo

O objetivo da análise é investigar como fatores socioeconômicos e demográficos estão associados às taxas de homicídio nos municípios paulistas.

Entre os principais fatores analisados estão:

- Urbanização
- Mortalidade infantil
- Desemprego
- PIB per capita
- Desigualdade de renda (Índice de Gini)
- Estrutura etária e proporção de homens jovens
- População municipal

A análise utiliza técnicas de suavização estatística, modelagem preditiva e análise espacial para produzir estimativas mais robustas das taxas de homicídio.

---

## Base de dados

A análise utiliza indicadores socioeconômicos municipais combinados com dados de homicídios para os municípios do estado de São Paulo.

Principais variáveis utilizadas:

- população residente
- taxa de mortalidade infantil
- taxa de urbanização
- taxa de desemprego
- PIB per capita
- índice de Gini
- proporção de homens jovens na população

Dados brutos
     ↓
Tratamento de dados
     ↓
Taxa de homicídios + Empirical Bayes
     ↓
Análise exploratória
     ↓
Diagnósticos (VIF, Cook's distance)
     ↓
Modelagem
     ↓
Avaliação
     ↓
Interpretabilidade (SHAP)
     ↓
Análise espacial (Moran's I)

---

# Estrutura do Projeto
project
│
├── data
│ ├── raw
│ └── geodata
│
├── outputs
│ ├── figures
│ └── tables
│
├── src
│ ├── data_processing.py
│ ├── data_quality.py
│ ├── data_comparison.py
│ ├── exploratory.py
│ ├── bayes_estimator.py
│ ├── modeling.py
│ ├── evaluation.py
│ ├── diagnostics.py
│ ├── influence.py
│ ├── model_explainability.py
│ ├── pca_analysis.py
│ ├── spatial_analysis.py
│ └── visualization.py
│
├── analysis.ipynb
├── main.py
├── run_pipeline.py
├── requirements.txt
└── README.md


---

# Pipeline de Análise

A pipeline executa automaticamente todas as etapas do processo analítico.

As principais etapas incluem:

### 1. Carregamento dos dados

Os dados são carregados a partir de um dataset contendo indicadores socioeconômicos e registros de homicídios municipais.

A leitura é realizada com tratamento automático de variáveis numéricas. 

---

### 2. Avaliação da qualidade dos dados

É gerado um relatório de qualidade contendo:

- tipo das variáveis
- quantidade de valores ausentes
- porcentagem de missing
- número de valores únicos
- estatísticas descritivas

Esse relatório é salvo em:
outputs/tables/data_quality_report.xlsx


A função responsável está implementada em `data_quality.py`. 

---

### 3. Tratamento dos dados

Algumas regras de tratamento são aplicadas:

- homicídios ausentes são considerados como zero
- mortalidade infantil é imputada pela mediana
- valores negativos de desemprego são corrigidos para zero

Essas etapas são executadas em `data_processing.py`. 

Também é gerado um relatório comparando os dados antes e depois do tratamento. 

---

### 4. Engenharia de variáveis

Uma nova variável é criada para representar a proporção estimada de homens jovens na população:
per_cent_young_male = per_cent_male × per_cent_youth


Essa variável busca capturar efeitos demográficos associados à criminalidade.

---

### 5. Cálculo da taxa de homicídios

A taxa de homicídios é calculada por 100 mil habitantes.

Além da taxa observada, também é aplicada uma **suavização Empirical Bayes** para reduzir a variabilidade em municípios com pequenas populações. 

`bayes_estimator.py`
---

### 6. Transformações logarítmicas

Algumas variáveis são transformadas em escala log para reduzir assimetria: `scenarios.py`

- taxa de homicídios suavizada
- PIB per capita
- população municipal

---

### 7. Estatísticas descritivas

É gerada uma tabela contendo:

- média
- desvio padrão
- mínimo
- máximo

para as variáveis analisadas: `exploratory.py`

---

### 8. Visualizações exploratórias

A pipeline gera diversos gráficos exploratórios:

- distribuição das taxas de homicídio
- comparação entre taxa observada e suavizada
- funil de variabilidade
- matriz de correlação
- diagnóstico da transformação log
- mapa espacial das taxas de homicídio

Essas visualizações são implementadas em `visualization.py`.

---

### 9. Diagnósticos estatísticos

São realizadas análises para avaliar propriedades dos dados e dos modelos.

Incluem:

**Cook's Distance**

Identifica observações influentes na regressão: `influence.py`

**VIF (Variance Inflation Factor)**

Avalia multicolinearidade entre variáveis explicativas: `diagnostics.py`

---

### 10. Modelagem

Três modelos são estimados:

- Regressão Linear (OLS)
- Random Forest
- XGBoost

A divisão entre treino e teste é realizada utilizando validação hold-out. 

Os modelos são avaliados utilizando:

- R²
- RMSE

Os resultados são consolidados automaticamente em tabelas comparativas. 

---

### 11. Interpretabilidade dos modelos

Para o modelo baseado em árvores é realizada análise de interpretabilidade utilizando **SHAP (SHapley Additive Explanations)**.

Essa técnica permite avaliar o impacto de cada variável nas previsões do modelo. 

---

### 12. Análise de Componentes Principais (PCA)

A PCA é utilizada para explorar padrões estruturais entre os indicadores socioeconômicos municipais. 

---

### 13. Análise espacial

A pipeline também realiza análise espacial utilizando o **Índice de Moran**, que mede autocorrelação espacial das taxas de homicídio. 

Essa análise permite verificar se municípios vizinhos apresentam padrões semelhantes de criminalidade.

---

# Resultados Gerados

A execução da pipeline gera automaticamente:

### Tabelas
outputs/tables/

Exemplos:

- relatório de qualidade dos dados
- estatísticas descritivas
- comparação de modelos
- análise de multicolinearidade

---

### Figuras
outputs/figures/


Exemplos:

- histogramas
- mapas espaciais
- funil de variabilidade
- gráficos de comparação de modelos
- análise SHAP
- PCA

---

## Como reproduzir o estudo

1. Clonar o repositório

git clone https://github.com/jackiemoliveira/criminalidade-sp-tcc.git

2. Criar ambiente virtual

python -m venv venv

3. Instalar dependências
pip install -r requirements.txt


As principais bibliotecas utilizadas incluem:

- pandas
- numpy
- scikit-learn
- statsmodels
- xgboost
- geopandas
- shap
- seaborn


4. Executar a pipeline
python main.py


A execução completa gera automaticamente todas as tabelas e figuras da análise. :contentReference[oaicite:14]{index=14}

---

# Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- XGBoost
- SHAP
- GeoPandas
- Matplotlib
- Seaborn

---

# Estrutura Analítica

A pipeline segue um fluxo estruturado de análise de dados:

1. ingestão de dados  
2. avaliação de qualidade  
3. tratamento e limpeza  
4. engenharia de variáveis  
5. análise exploratória  
6. diagnóstico estatístico  
7. modelagem preditiva  
8. interpretabilidade  
9. análise espacial  

---

# Autor
- Jaqueline de Moura
- Projeto desenvolvido para estudo sobre determinantes socioeconômicos da criminalidade municipal no estado de São Paulo.
