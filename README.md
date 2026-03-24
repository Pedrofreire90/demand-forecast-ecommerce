# 📦 Previsão de Demanda para E-commerce

> Análise de tendências e previsão de vendas com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-Data-green)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

## 🎯 Objetivo

Este projeto simula um cenário real de e-commerce brasileiro e constrói um pipeline completo de **previsão de demanda** — desde a análise exploratória até a previsão dos próximos 30 dias — com o objetivo de auxiliar decisões de **gestão de estoque e planejamento comercial**.

---

## 📊 Resultados do Modelo

| Métrica | Resultado |
|---|---|
| MAE (Erro Absoluto Médio) | **~20 unidades** |
| MAPE (Erro % Médio) | **~9.8%** |
| R² (Coef. de Determinação) | **0.85** |

> O modelo explica **85% da variação** nas vendas diárias, com erro médio de menos de 10%.

---

## 🛠️ Tecnologias

- **Python 3.10+**
- **pandas** — manipulação e análise de dados
- **scikit-learn** — `RandomForestRegressor`, `TimeSeriesSplit`, métricas
- **matplotlib / seaborn** — visualizações
- **numpy** — operações numéricas

---

## 📁 Estrutura do Projeto

```
demand_forecast/
├── forecast.py        # Pipeline completo (EDA → modelo → previsão)
├── README.md
└── outputs/
    ├── 1_eda_analise.png        # Análise exploratória
    ├── 2_resultado_modelo.png   # Real vs Previsto
    └── 3_insights_previsao.png  # Importância das features + previsão futura
```

---

## 🔄 Pipeline do Projeto

### 1. Geração do Dataset
Dataset simulado com **730 dias de vendas** contendo:
- Tendência de crescimento anual (~20%)
- Sazonalidade semanal (fins de semana vendem ~35% mais)
- Datas comemorativas brasileiras (Black Friday, Natal, Dia das Mães...)
- Promoções aleatórias (flash sales)
- Ruído gaussiano realista

### 2. Análise Exploratória (EDA)
- Série temporal com média móvel de 30 dias
- Distribuição de vendas por dia da semana
- Boxplot mensal (sazonalidade anual)
- Comparativo 2022 vs 2023

### 3. Feature Engineering
Criação de **20 variáveis preditoras**:

| Categoria | Features |
|---|---|
| Calendário | dia_semana, mês, trimestre, semana_ano |
| Flags | é_fim_de_semana, é_dezembro, é_novembro, promoção |
| Lags | vendas 1d, 7d, 14d, 21d, 28d atrás |
| Médias móveis | média 7d, 14d, 28d |
| Volatilidade | desvio padrão 7d, 28d |

### 4. Modelo — Random Forest
- `RandomForestRegressor` com 200 árvores
- Validação com **TimeSeriesSplit** (respeita ordem cronológica)
- Split: 80% treino / 20% teste

### 5. Previsão Futura
- Previsão iterativa para os **próximos 30 dias**
- Utiliza previsões anteriores como novos lags (rolling forecast)
- Intervalo de confiança de ±10%

---

## 🚀 Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/Pedrofreire90/demand-forecast-ecommerce
cd demand-forecast-ecommerce

# 2. Instale as dependências
pip install pandas scikit-learn matplotlib seaborn numpy

# 3. Execute
python forecast.py
```

---

## 📈 Visualizações

### EDA — Análise Exploratória
Análise de sazonalidade semanal, mensal e tendência anual de crescimento.

### Real vs Previsto
Comparação do modelo com dados reais no período de teste, com destaque para o erro (área sombreada).

### Importância das Features + Previsão 30 dias
Quais variáveis mais influenciam as vendas e a projeção para o próximo mês.

---

## 💡 Aplicações Reais

Este projeto pode ser adaptado para:
- **Gestão de estoque** — evitar ruptura ou excesso de produto
- **Planejamento de compras** — negociação com fornecedores
- **Campanhas promocionais** — identificar períodos de baixa demanda
- **Logística** — dimensionamento de capacidade de entrega

---

## 👨‍💻 Autor

**Pedro Freire**  
Python Developer | Automações & Bots  
[![GitHub](https://img.shields.io/badge/GitHub-Pedrofreire90-black?logo=github)](https://github.com/Pedrofreire90)

---

*Projeto desenvolvido para portfólio — aberto a sugestões e colaborações.*
