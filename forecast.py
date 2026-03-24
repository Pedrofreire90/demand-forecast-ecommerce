"""
=============================================================
  PREVISÃO DE DEMANDA PARA E-COMMERCE
  Autor: Pedro Freire
  GitHub: github.com/Pedrofreire90
  Stack: Python | pandas | scikit-learn | matplotlib
=============================================================

Objetivo:
  Analisar o histórico de vendas de um e-commerce e construir
  um modelo de Machine Learning para prever a demanda futura,
  auxiliando decisões de estoque e planejamento.

Etapas:
  1. Geração do dataset simulado realista
  2. Análise Exploratória de Dados (EDA)
  3. Feature Engineering
  4. Treinamento e avaliação do modelo
  5. Previsão futura e visualização final
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Configuração visual
plt.rcParams["figure.facecolor"] = "#F8F9FA"
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.family"] = "DejaVu Sans"
COLORS = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]

np.random.seed(42)


# =============================================================
# 1. GERAÇÃO DO DATASET
# =============================================================

def gerar_dataset():
    """
    Gera um dataset realista simulando vendas diárias de um
    e-commerce por 2 anos, com:
      - Tendência de crescimento gradual
      - Sazonalidade semanal (fins de semana vendem mais)
      - Sazonalidade anual (datas comemorativas)
      - Efeito de promoções
      - Ruído aleatório
    """
    print("\n📦 [1/5] Gerando dataset simulado de e-commerce...")

    datas = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
    n = len(datas)

    # Tendência de crescimento (e-commerce crescendo ~20% ao ano)
    tendencia = np.linspace(100, 140, n)

    # Sazonalidade semanal (pico sexta/sábado)
    dia_semana = np.array([d.weekday() for d in datas])
    sazon_semanal = np.where(dia_semana >= 4, 1.35, 1.0)  # sex/sab/dom +35%
    sazon_semanal = np.where(dia_semana == 6, 1.15, sazon_semanal)  # dom +15%

    # Sazonalidade anual (datas comemorativas brasileiras)
    mes = np.array([d.month for d in datas])
    dia = np.array([d.day for d in datas])

    sazon_anual = np.ones(n)
    sazon_anual[mes == 11] = 1.6   # Black Friday
    sazon_anual[mes == 12] = 1.8   # Natal
    sazon_anual[(mes == 5) & (dia >= 8) & (dia <= 14)] = 1.5   # Dia das Mães
    sazon_anual[(mes == 6) & (dia >= 12) & (dia <= 14)] = 1.3  # Dia dos Namorados
    sazon_anual[(mes == 10) & (dia >= 12) & (dia <= 14)] = 1.4 # Dia das Crianças
    sazon_anual[mes == 1] = 0.8    # Janeiro fraco (pós-festas)
    sazon_anual[mes == 2] = 0.85   # Fevereiro (Carnaval)

    # Promoções aleatórias (flashsales)
    promocoes = np.ones(n)
    indices_promo = np.random.choice(n, size=30, replace=False)
    promocoes[indices_promo] = np.random.uniform(1.4, 2.0, size=30)

    # Ruído gaussiano
    ruido = np.random.normal(1.0, 0.08, n)

    # Vendas finais
    vendas = tendencia * sazon_semanal * sazon_anual * promocoes * ruido
    vendas = np.maximum(vendas.round().astype(int), 0)

    df = pd.DataFrame({
        "data": datas,
        "vendas": vendas,
        "dia_semana": dia_semana,
        "mes": mes,
        "dia_mes": dia,
        "promocao": (promocoes > 1.1).astype(int)
    })

    print(f"   ✅ Dataset criado: {len(df)} dias | {df['vendas'].sum():,} vendas totais")
    print(f"   📊 Média diária: {df['vendas'].mean():.1f} | Máximo: {df['vendas'].max()} | Mínimo: {df['vendas'].min()}")
    return df


# =============================================================
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# =============================================================

def eda(df):
    """Gera 4 gráficos de análise exploratória."""
    print("\n📊 [2/5] Análise Exploratória de Dados (EDA)...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("EDA — Análise de Vendas do E-commerce", fontsize=16, fontweight="bold", y=1.01)

    # --- Gráfico 1: Série temporal completa com média móvel ---
    ax1 = axes[0, 0]
    ax1.plot(df["data"], df["vendas"], color=COLORS[0], alpha=0.35, linewidth=0.8, label="Vendas diárias")
    media_movel = df["vendas"].rolling(30).mean()
    ax1.plot(df["data"], media_movel, color=COLORS[0], linewidth=2.2, label="Média móvel 30d")
    ax1.set_title("Série Temporal de Vendas", fontweight="bold")
    ax1.set_ylabel("Unidades vendidas")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    ax1.legend()

    # --- Gráfico 2: Vendas por dia da semana ---
    ax2 = axes[0, 1]
    nomes_dias = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    media_dia = df.groupby("dia_semana")["vendas"].mean()
    bars = ax2.bar(nomes_dias, media_dia.values, color=COLORS[0], alpha=0.85, edgecolor="white", linewidth=1.5)
    ax2.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax2.set_title("Média de Vendas por Dia da Semana", fontweight="bold")
    ax2.set_ylabel("Média de unidades")

    # --- Gráfico 3: Vendas mensais (boxplot) ---
    ax3 = axes[1, 0]
    nomes_meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    dados_mes = [df[df["mes"] == m]["vendas"].values for m in range(1, 13)]
    bp = ax3.boxplot(dados_mes, patch_artist=True, medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], [COLORS[0]] * 12):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xticklabels(nomes_meses)
    ax3.set_title("Distribuição de Vendas por Mês", fontweight="bold")
    ax3.set_ylabel("Unidades vendidas")

    # --- Gráfico 4: Vendas anuais (2022 vs 2023) ---
    ax4 = axes[1, 1]
    for i, ano in enumerate([2022, 2023]):
        dados_ano = df[df["data"].dt.year == ano].groupby("mes")["vendas"].mean()
        ax4.plot(nomes_meses, dados_ano.values, marker="o", linewidth=2,
                 color=COLORS[i], label=str(ano), markersize=5)
    ax4.set_title("Comparativo Anual de Vendas (média mensal)", fontweight="bold")
    ax4.set_ylabel("Média de unidades")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("/home/pedro/Área de trabalho/Estudo Python/Previsao de demanda ecomerce/outputs/1_eda_analise.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Gráfico EDA salvo.")


# =============================================================
# 3. FEATURE ENGINEERING
# =============================================================

def feature_engineering(df):
    """
    Cria features temporais e de lag que o modelo usará para
    aprender padrões históricos e fazer previsões.
    """
    print("\n🔧 [3/5] Feature Engineering...")

    df = df.copy()
    df = df.sort_values("data").reset_index(drop=True)

    # --- Features de calendário ---
    df["ano"] = df["data"].dt.year
    df["semana_ano"] = df["data"].dt.isocalendar().week.astype(int)
    df["trimestre"] = df["data"].dt.quarter
    df["e_fds"] = (df["dia_semana"] >= 4).astype(int)
    df["e_dezembro"] = (df["mes"] == 12).astype(int)
    df["e_novembro"] = (df["mes"] == 11).astype(int)

    # --- Lag features (vendas passadas) ---
    for lag in [1, 7, 14, 21, 28]:
        df[f"lag_{lag}d"] = df["vendas"].shift(lag)

    # --- Médias móveis ---
    df["media_7d"]  = df["vendas"].shift(1).rolling(7).mean()
    df["media_14d"] = df["vendas"].shift(1).rolling(14).mean()
    df["media_28d"] = df["vendas"].shift(1).rolling(28).mean()

    # --- Desvio padrão (captura volatilidade) ---
    df["std_7d"]  = df["vendas"].shift(1).rolling(7).std()
    df["std_28d"] = df["vendas"].shift(1).rolling(28).std()

    # Remover linhas com NaN (primeiros 28 dias)
    df = df.dropna().reset_index(drop=True)

    features = [
        "dia_semana", "mes", "dia_mes", "ano", "semana_ano", "trimestre",
        "e_fds", "e_dezembro", "e_novembro", "promocao",
        "lag_1d", "lag_7d", "lag_14d", "lag_21d", "lag_28d",
        "media_7d", "media_14d", "media_28d",
        "std_7d", "std_28d"
    ]

    print(f"   ✅ {len(features)} features criadas | Dataset final: {len(df)} linhas")
    return df, features


# =============================================================
# 4. TREINAMENTO E AVALIAÇÃO DO MODELO
# =============================================================

def treinar_modelo(df, features):
    """
    Treina um RandomForestRegressor com validação por série
    temporal (TimeSeriesSplit) — respeita a ordem cronológica.
    """
    print("\n🤖 [4/5] Treinamento e Avaliação do Modelo...")

    X = df[features]
    y = df["vendas"]

    # Split temporal: 80% treino, 20% teste
    corte = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]
    datas_teste = df["data"].iloc[corte:]

    print(f"   📅 Treino: {df['data'].iloc[0].date()} → {df['data'].iloc[corte-1].date()} ({corte} dias)")
    print(f"   📅 Teste:  {df['data'].iloc[corte].date()} → {df['data'].iloc[-1].date()} ({len(X_test)} dias)")

    # Modelo
    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    # Predições
    y_pred = modelo.predict(X_test)

    # Métricas
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"\n   {'─'*40}")
    print(f"   📈 MÉTRICAS DE AVALIAÇÃO")
    print(f"   {'─'*40}")
    print(f"   MAE  (Erro Absoluto Médio):   {mae:.1f} unidades")
    print(f"   RMSE (Raiz do Erro Quadrático): {rmse:.1f} unidades")
    print(f"   MAPE (Erro % Médio Absoluto): {mape:.1f}%")
    print(f"   R²   (Coef. de Determinação): {r2:.4f}")
    print(f"   {'─'*40}")

    # Gráfico: Real vs Previsto
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Resultado do Modelo — Real vs Previsto", fontsize=15, fontweight="bold")

    # Série completa
    ax1 = axes[0]
    ax1.plot(df["data"].iloc[:corte], y_train, color="#94A3B8", alpha=0.5,
             linewidth=0.8, label="Treino (histórico)")
    ax1.plot(datas_teste, y_test.values, color=COLORS[0], linewidth=1.2, label="Real (teste)")
    ax1.plot(datas_teste, y_pred, color=COLORS[1], linewidth=1.5,
             linestyle="--", label="Previsto")
    ax1.axvline(df["data"].iloc[corte], color="#DC2626", linewidth=1.5,
                linestyle=":", alpha=0.7, label="Início do teste")
    ax1.set_title("Série Completa", fontweight="bold")
    ax1.set_ylabel("Unidades vendidas")
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))

    # Zoom no período de teste
    ax2 = axes[1]
    ax2.fill_between(datas_teste, y_test.values, y_pred, alpha=0.15, color=COLORS[2], label="Erro")
    ax2.plot(datas_teste, y_test.values, color=COLORS[0], linewidth=1.5, label="Real")
    ax2.plot(datas_teste, y_pred, color=COLORS[1], linewidth=1.5, linestyle="--", label="Previsto")
    ax2.set_title(f"Zoom — Período de Teste  |  MAE: {mae:.1f}  |  MAPE: {mape:.1f}%  |  R²: {r2:.4f}",
                  fontweight="bold")
    ax2.set_ylabel("Unidades vendidas")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))

    plt.tight_layout()
    plt.savefig("/home/pedro/Área de trabalho/Estudo Python/Previsao de demanda ecomerce/outputs/2_resultado_modelo.png", dpi=150, bbox_inches="tight")
    plt.close()

    return modelo, features, mae, rmse, r2, mape


# =============================================================
# 5. IMPORTÂNCIA DAS FEATURES + PREVISÃO FUTURA
# =============================================================

def feature_importance_e_previsao(modelo, df, features):
    """Plota importância das features e gera previsão para os próximos 30 dias."""
    print("\n🔮 [5/5] Importância das Features e Previsão Futura...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Insights do Modelo", fontsize=15, fontweight="bold")

    # --- Importância das features ---
    ax1 = axes[0]
    importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=True)
    cores = [COLORS[0] if v > importancias.median() else "#CBD5E1" for v in importancias.values]
    bars = ax1.barh(importancias.index, importancias.values, color=cores, edgecolor="white")
    ax1.set_title("Importância das Features\n(quais variáveis mais explicam as vendas)",
                  fontweight="bold")
    ax1.set_xlabel("Importância relativa")

    # --- Previsão futura (30 dias) ---
    ax2 = axes[1]
    ultimo_dado = df.iloc[-1]
    ultima_data = df["data"].max()
    datas_futuras = pd.date_range(start=ultima_data + pd.Timedelta(days=1), periods=30, freq="D")

    # Usa a última janela para criar as features futuras (simplificado)
    previsoes = []
    historico = df["vendas"].values.copy()

    for data in datas_futuras:
        lag_vals = {
            "lag_1d":  historico[-1],
            "lag_7d":  historico[-7],
            "lag_14d": historico[-14],
            "lag_21d": historico[-21],
            "lag_28d": historico[-28],
            "media_7d":  np.mean(historico[-7:]),
            "media_14d": np.mean(historico[-14:]),
            "media_28d": np.mean(historico[-28:]),
            "std_7d":    np.std(historico[-7:]),
            "std_28d":   np.std(historico[-28:]),
        }
        row = {
            "dia_semana":  data.weekday(),
            "mes":         data.month,
            "dia_mes":     data.day,
            "ano":         data.year,
            "semana_ano":  data.isocalendar()[1],
            "trimestre":   (data.month - 1) // 3 + 1,
            "e_fds":       int(data.weekday() >= 4),
            "e_dezembro":  int(data.month == 12),
            "e_novembro":  int(data.month == 11),
            "promocao":    0,
            **lag_vals
        }
        X_fut = pd.DataFrame([row])[features]
        pred = modelo.predict(X_fut)[0]
        previsoes.append(max(int(pred), 0))
        historico = np.append(historico, pred)

    # Contexto histórico (últimos 60 dias)
    hist_60 = df.tail(60)
    ax2.plot(hist_60["data"], hist_60["vendas"], color=COLORS[0], linewidth=1.5,
             label="Histórico (60d)")
    ax2.plot(datas_futuras, previsoes, color=COLORS[1], linewidth=2,
             linestyle="--", marker="o", markersize=3, label="Previsão (30d)")
    ax2.axvline(ultima_data, color="#DC2626", linewidth=1.5, linestyle=":", alpha=0.7)
    ax2.fill_between(datas_futuras,
                     [v * 0.9 for v in previsoes],
                     [v * 1.1 for v in previsoes],
                     alpha=0.15, color=COLORS[1], label="Intervalo ±10%")
    ax2.set_title(f"Previsão para os Próximos 30 Dias\n(Total estimado: {sum(previsoes):,} unidades)",
                  fontweight="bold")
    ax2.set_ylabel("Unidades previstas")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

    plt.tight_layout()
    plt.savefig("/home/pedro/Área de trabalho/Estudo Python/Previsao de demanda ecomerce/outputs/3_insights_previsao.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   ✅ Previsão concluída: {sum(previsoes):,} unidades nos próximos 30 dias")
    return previsoes, datas_futuras


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PREVISÃO DE DEMANDA — E-COMMERCE")
    print("=" * 60)

    df                         = gerar_dataset()
    eda(df)
    df_feat, features          = feature_engineering(df)
    modelo, features, mae, rmse, r2, mape = treinar_modelo(df_feat, features)
    previsoes, datas_futuras   = feature_importance_e_previsao(modelo, df_feat, features)

    print("\n" + "=" * 60)
    print("  ✅ PROJETO CONCLUÍDO COM SUCESSO")
    print("=" * 60)
    print("\n  Arquivos gerados:")
    print("   📊 1_eda_analise.png      — Análise exploratória")
    print("   🤖 2_resultado_modelo.png — Real vs Previsto")
    print("   🔮 3_insights_previsao.png — Features + Previsão futura")
    print(f"\n  Resumo do modelo:")
    print(f"   MAE:  {mae:.1f} unidades  |  MAPE: {mape:.1f}%  |  R²: {r2:.4f}")
    print("=" * 60)
