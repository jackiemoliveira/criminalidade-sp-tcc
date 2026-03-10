import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



sns.set(style="whitegrid")
# ---------------------------------------------------
# Clusterização de municípios por perfil socioeconômico
# ---------------------------------------------------

def plot_municipality_clusters(df, features, output_path, k=4):

    data = df[features].dropna()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=k, random_state=42)

    clusters = kmeans.fit_predict(X_scaled)

    df_cluster = data.copy()

    df_cluster["cluster"] = clusters

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        x=df_cluster[features[0]],
        y=df_cluster[features[1]],
        hue=df_cluster["cluster"],
        palette="Set2"
    )

    plt.title("Clusterização de municípios por perfil socioeconômico")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

# ---------------------------------------------------
# Ranking de municípios mais violentos
# ---------------------------------------------------
def plot_top_violent_municipalities(df, output_path, top_n=15):

    ranking = (
        df.sort_values("tx_homicidios_bayes", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10,8))

    plt.barh(
        ranking["municipio"],
        ranking["tx_homicidios_bayes"]
    )

    plt.xlabel("Taxa de homicídios (por 100 mil habitantes)")
    plt.title(f"Top {top_n} municípios com maiores taxas de homicídio")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

# ---------------------------------------------------
# Ranking de municípios menos violentos
# ---------------------------------------------------
def plot_least_violent_municipalities(df, output_path, top_n=15):

    ranking = (
        df.sort_values("tx_homicidios_bayes", ascending=True)
        .head(top_n)
    )

    plt.figure(figsize=(10,8))

    plt.barh(
        ranking["municipio"],
        ranking["tx_homicidios_bayes"]
    )

    plt.xlabel("Taxa de homicídios (por 100 mil habitantes)")
    plt.title(f"Top {top_n} municípios com menores taxas de homicídio")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

# ---------------------------------------------------
# Heatmap correlação entre variáveis
# ---------------------------------------------------

def plot_correlation_heatmap(df, variables, output_path):

    corr = df[variables].corr()

    plt.figure(figsize=(10,8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Matriz de Correlação entre Variáveis Socioeconômicas")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

# ---------------------------------------------------
# Histograma da Taxa de Homicídios
# ---------------------------------------------------

def plot_histogram(df, output_path):

    plt.figure(figsize=(10,6))

    sns.histplot(
        df["tx_homicidios_bayes"],
        bins=25,
        kde=True
    )

    plt.title(
        "Distribuição da Taxa de Homicídios nos Municípios de São Paulo"
    )

    plt.xlabel(
        "Taxa de homicídios (por 100 mil habitantes)"
    )

    plt.ylabel("Frequência")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()



# ---------------------------------------------------
# Funil de variabilidade
# ---------------------------------------------------

def plot_funnel(df, population_col, rate_col, output_path):

    pop = df[population_col].values
    rate = df[rate_col].values

    taxa_media = np.mean(rate)

    # sequência suave de populações
    pop_seq = np.linspace(pop.min(), pop.max(), 500)

    # eventos esperados
    expected = taxa_media * pop_seq / 100000

    # intervalo Poisson
    limite_sup_event = expected + 1.96 * np.sqrt(expected)
    limite_inf_event = expected - 1.96 * np.sqrt(expected)

    limite_inf_event[limite_inf_event < 0] = 0

    # converter para taxa
    limite_sup = limite_sup_event / pop_seq * 100000
    limite_inf = limite_inf_event / pop_seq * 100000

    plt.figure(figsize=(9,6))

    # pontos dos municípios
    plt.scatter(
        pop,
        rate,
        alpha=0.6,
        label="Municípios"
    )

    # limites do funil
    plt.plot(
        pop_seq,
        limite_sup,
        color="red",
        linestyle="--",
        label="Limite superior (95%)"
    )

    plt.plot(
        pop_seq,
        limite_inf,
        color="red",
        linestyle="--",
        label="Limite inferior (95%)"
    )

    # média estadual
    plt.axhline(
        taxa_media,
        color="black",
        linestyle="-",
        label="Média estadual"
    )

    plt.xscale("log")

    plt.xlabel("População do município (escala log)")
    plt.ylabel("Taxa de homicídios observada")

    plt.title("Funil de variabilidade das taxas de homicídios municipais")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()



# ---------------------------------------------------
# Importância das variáveis
# ---------------------------------------------------

def plot_feature_importance(
        model,
        feature_names,
        title,
        output_path
):

    importances = model.feature_importances_

    indices = np.argsort(importances)

    plt.figure(figsize=(8,6))

    plt.barh(
        np.array(feature_names)[indices],
        importances[indices]
    )

    plt.title(title)

    plt.xlabel("Importância")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()



# ---------------------------------------------------
# Cook's distance
# ---------------------------------------------------

def plot_cooks_distance(
        df,
        output_path
):

    n = len(df)

    threshold = 4 / n

    plt.figure(figsize=(10,6))

    plt.scatter(
        range(n),
        df["cooks_distance"]
    )

    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label="Threshold (4/n)"
    )

    plt.xlabel("Observação")

    plt.ylabel("Cook's Distance")

    plt.title(
        "Cook's Distance - Observações Influentes"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()



# ---------------------------------------------------
# Observado vs Bayes
# ---------------------------------------------------

def plot_observed_vs_bayes(
        df,
        output_path
):

    plt.figure(figsize=(8,6))

    plt.scatter(
        df["tx_homicidios"],
        df["tx_homicidios_bayes"],
        alpha=0.6
    )

    plt.xlabel(
        "Taxa de homicídios observada"
    )

    plt.ylabel(
        "Taxa suavizada (Empirical Bayes)"
    )

    plt.title(
        "Comparação entre taxa observada e taxa suavizada de homicídios"
    )

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()



# ---------------------------------------------------
# Distribuição observada vs suavizada
# ---------------------------------------------------

def plot_distribution_comparison(
        df,
        output_path
):

    plt.figure(figsize=(10,6))

    sns.histplot(
        df["tx_homicidios"],
        color="steelblue",
        label="Observada",
        bins=30,
        alpha=0.5
    )

    sns.histplot(
        df["tx_homicidios_bayes"],
        color="orange",
        label="Suavizada (Bayes)",
        bins=30,
        alpha=0.6
    )

    plt.legend()

    plt.xlabel(
        "Taxa de homicídios"
    )

    plt.ylabel("Frequência")

    plt.title(
        "Distribuição das taxas de homicídios: observada vs suavizada"
    )

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()



# ---------------------------------------------------
# taxa de homicídios vs população
# ---------------------------------------------------

def plot_rate_vs_population(
        df,
        output_path
):

    plt.figure(figsize=(8,6))

    plt.scatter(
        df["resident_population"],
        df["tx_homicidios"],
        alpha=0.6
    )

    plt.xlabel("População do município")

    plt.ylabel("Taxa de homicídios observada")

    plt.title(
        "Taxa de homicídios observada vs população municipal"
    )

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()



# ---------------------------------------------------
# taxa de Bayes vs população
# ---------------------------------------------------

def plot_bayes_vs_population(
        df,
        output_path
):

    plt.figure(figsize=(8,6))

    plt.scatter(
        df["resident_population"],
        df["tx_homicidios_bayes"],
        alpha=0.6
    )

    plt.xlabel("População do município")

    plt.ylabel(
        "Taxa de homicídios suavizada (Bayes)"
    )

    plt.title(
        "Taxa de homicídios suavizada vs população municipal"
    )

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()



# ---------------------------------------------------
# taxa de homicídios vs população em escala log
# ---------------------------------------------------

def plot_rate_vs_population_log(
        df,
        output_path
):

    plt.figure(figsize=(8,6))

    plt.scatter(
        df["resident_population"],
        df["tx_homicidios"],
        alpha=0.6
    )

    plt.xscale("log")

    plt.xlabel(
        "População do município (escala log)"
    )

    plt.ylabel(
        "Taxa de homicídios observada"
    )

    plt.title(
        "Taxa de homicídios observada vs população (escala log)"
    )

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()



# ---------------------------------------------------
# Mapa espacial
# ---------------------------------------------------
import geopandas as gpd
import matplotlib.pyplot as plt

def plot_spatial_distribution(
        df,
        geo_path,
        output_path
):
    
    gdf = gpd.read_file(geo_path)

    # garantir tipo igual
    df["codigo"] = df["codigo"].astype(str)
    gdf["id"] = gdf["id"].astype(str)

    # merge usando código IBGE
    gdf = gdf.merge(
        df,
        left_on="id",
        right_on="codigo",
        how="left"
    )

    fig, ax = plt.subplots(figsize=(12,10))

    gdf.plot(
        column="tx_homicidios_bayes",
        cmap="Reds",
        legend=True,
        ax=ax,
        edgecolor="black",
        linewidth=0.2
    )

    ax.set_title(
        "Distribuição Espacial da Taxa de Homicídios Bayes\nMunicípios do Estado de São Paulo"
    )

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

# ---------------------------------------------------
# Comparação da variável resposta Bayes vs Bayes em escala log
# ---------------------------------------------------
    
def plot_log_diagnostic(df, output_path):

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10,5))

    # Distribuição original
    plt.subplot(1,2,1)

    sns.histplot(
        df["tx_homicidios_bayes"],
        bins=30,
        kde=True
    )

    plt.title("Distribuição da taxa de homicídios")

    # Distribuição log
    plt.subplot(1,2,2)

    sns.histplot(
        df["log_tx_homicidios_bayes"],
        bins=30,
        kde=True
    )

    plt.title("Distribuição após transformação log")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

# ---------------------------------------------------
# Comparação entre modelos
# ---------------------------------------------------
def plot_model_comparison(results_df, output_path):

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=results_df,
        x="modelo",
        y="R2",
        hue="dataset"
    )

    plt.title("Comparação do desempenho dos modelos")
    plt.ylabel("R²")
    plt.xlabel("Modelo")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()