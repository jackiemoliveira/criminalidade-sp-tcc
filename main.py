import os
import numpy as np
import pandas as pd

from src.config import TEST_SIZE, RANDOM_STATE

from src.data_processing import load_data, clean_data
from src.data_quality import data_quality_report
from src.data_comparison import compare_original_vs_treated

from src.bayes_estimator import calculate_bayes_rate
from src.exploratory import descriptive_statistics

from src.influence import cooks_distance_analysis
from src.diagnostics import calculate_vif

from src.modeling import train_models
from src.evaluation import (
    evaluate_models,
    create_results_table
)

from src.visualization import (
    plot_histogram,
    plot_observed_vs_bayes,
    plot_distribution_comparison,
    plot_funnel,
    plot_correlation_heatmap,
    plot_spatial_distribution,
    plot_cooks_distance,
    plot_log_diagnostic,
    plot_model_comparison
)

from src.model_explainability import shap_analysis
from src.pca_analysis import run_pca_analysis
from src.spatial_analysis import calculate_morans_i


def main():

    print("Iniciando pipeline")

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # =====================================================
    # 1. CARREGAR DADOS
    # =====================================================

    df = load_data("data/raw/db_homicidios_sp_onu.csv")

    # =====================================================
    # 2. RELATÓRIO DE QUALIDADE
    # =====================================================

    quality = data_quality_report(df)

    quality.to_excel(
        "outputs/tables/data_quality_report.xlsx",
        index=False
    )

    # =====================================================
    # 3. TRATAMENTO
    # =====================================================

    df_original, df = clean_data(df)

    variaveis_base = [
        "homicide_count",
        "per_cent_urban",
        "resident_population",
        "infant_mortality",
        "per_cent_male",
        "per_cent_youth",
        "per_cent_unemployed",
        "gdp_per_capita",
        "gini"
    ]

    comparacao = compare_original_vs_treated(
        df_original,
        df,
        variaveis_base
    )

    comparacao.to_excel(
        "outputs/tables/comparacao_tratamento_dados.xlsx",
        index=False
    )

    # =====================================================
    # 4. FEATURE ENGINEERING
    # =====================================================

    df["per_cent_young_male"] = (
        df["per_cent_male"] * df["per_cent_youth"]
    ) / 100

    # =====================================================
    # 5. TAXA DE HOMICÍDIOS
    # =====================================================

    df["tx_homicidios"] = (
        df["homicide_count"] /
        df["resident_population"]
    ) * 100000

    df = calculate_bayes_rate(df)

    # =====================================================
    # 6. TRANSFORMAÇÕES LOG
    # =====================================================

    df["log_tx_homicidios_bayes"] = np.log1p(df["tx_homicidios_bayes"])
    df["log_pib"] = np.log1p(df["gdp_per_capita"])
    df["log_populacao"] = np.log1p(df["resident_population"])

    # =====================================================
    # 7. ESTATÍSTICA DESCRITIVA
    # =====================================================

    tabela_desc = descriptive_statistics(df, variaveis_base)

    tabela_desc.to_excel(
        "outputs/tables/tabela_descritiva.xlsx"
    )

    # =====================================================
    # 8. VISUALIZAÇÕES
    # =====================================================

    plot_histogram(
        df,
        "outputs/figures/histograma_taxa_homicidios.png"
    )

    plot_observed_vs_bayes(
        df,
        "outputs/figures/taxa_observada_vs_bayes.png"
    )

    plot_distribution_comparison(
        df,
        "outputs/figures/distribuicao_observada_vs_bayes.png"
    )

    plot_funnel(
        df,
        population_col="resident_population",
        rate_col="tx_homicidios_bayes",
        output_path="outputs/figures/funil_variabilidade.png"
    )

    plot_log_diagnostic(
        df,
        "outputs/figures/log_diagnostic.png"
    )

    plot_spatial_distribution(
        df,
        "data/geodata/municipios_sp.geojson",
        "outputs/figures/mapa_taxa_homicidios_sp.png"
    )

    # =====================================================
    # 9. DEFINIÇÃO DOS MODELOS
    # =====================================================

    features_sem_feature = [
        "per_cent_urban",
        "infant_mortality",
        "per_cent_unemployed",
        "log_pib",
        "gini",
        "log_populacao"
    ]

    features_com_feature = [
        "per_cent_urban",
        "infant_mortality",
        "per_cent_unemployed",
        "log_pib",
        "gini",
        "log_populacao",
        "per_cent_young_male"
    ]

    model_features = {
        "sem_feature": features_sem_feature,
        "com_homens_jovens": features_com_feature
    }

    targets = {
        "taxa": "tx_homicidios_bayes",
        "log": "log_tx_homicidios_bayes"
    }

    # =====================================================
    # 10. MATRIZ DE CORRELAÇÃO
    # =====================================================

    plot_correlation_heatmap(
        df,
        features_com_feature,
        "outputs/figures/matriz_correlacao.png"
    )

    # =====================================================
    # 11. COOK DISTANCE
    # =====================================================

    df = cooks_distance_analysis(
        df,
        features_com_feature,
        "tx_homicidios_bayes"
    )

    plot_cooks_distance(
        df,
        "outputs/figures/cooks_distance.png"
    )

    # =====================================================
    # 12. VIF
    # =====================================================

    calculate_vif(
        df,
        features_com_feature,
        "outputs/tables/vif_multicolinearidade.csv"
    )

    # =====================================================
    # 13. MODELAGEM
    # =====================================================

    resultados_finais = []

    for nome_target, target in targets.items():

        for nome_modelo, features in model_features.items():

            rf, xgb_model, ols, X_train, X_test, y_train, y_test = train_models(
                df,
                features,
                target
            )

            resultados = evaluate_models(
                rf,
                xgb_model,
                ols,
                X_train,
                X_test,
                y_train,
                y_test
            )

            resultados["target"] = nome_target
            resultados["cenario"] = nome_modelo

            resultados_finais.append(resultados)

    resultados_finais = pd.concat(resultados_finais, ignore_index=True)

    resultados_finais.to_excel(
        "outputs/tables/comparacao_modelos.xlsx",
        index=False
    )

    plot_model_comparison(
        resultados_finais,
        "outputs/figures/comparacao_modelos.png"
    )

    tabela_final = create_results_table(
    resultados_finais,
    "outputs/tables/tabela_resultados_modelos.xlsx"
    )
    
    # =====================================================
    # 14. SHAP
    # =====================================================

    shap_analysis(
        rf,
        X_test,
        "outputs/figures/shap_importance.png"
    )

    # =====================================================
    # 15. PCA
    # =====================================================

    run_pca_analysis(
        df,
        features_com_feature,
        "outputs/figures/pca_municipios.png"
    )

    # =====================================================
    # 16. MORAN
    # =====================================================

    calculate_morans_i(
        df,
        "data/geodata/municipios_sp.geojson",
        "outputs/figures/moran_scatterplot.png"
    )

    print("Pipeline finalizada com sucesso")


if __name__ == "__main__":
    main()