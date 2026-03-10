import statsmodels.api as sm
import pandas as pd
import numpy as np


def cooks_distance_analysis(df, features, target):

    df = df.copy()

    # garantir que todas variáveis existem
    features = [f for f in features if f in df.columns]

    if target not in df.columns:
        raise ValueError(f"Variável target '{target}' não encontrada no dataframe")

    # substituir infinitos
    df = df.replace([np.inf, -np.inf], np.nan)

    # selecionar colunas necessárias
    cols_model = features + [target]

    df_model = df[cols_model].dropna()

    # segurança caso dataframe fique vazio
    if df_model.shape[0] == 0:

        print("Aviso: nenhum dado disponível para cálculo do Cook's Distance")

        df["cooks_distance"] = np.nan

        return df

    X = df_model[features]
    y = df_model[target]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    influence = model.get_influence()

    cooks = influence.cooks_distance[0]

    df_model["cooks_distance"] = cooks

    # juntar resultado ao dataset original
    df = df.merge(
        df_model[["cooks_distance"]],
        left_index=True,
        right_index=True,
        how="left"
    )

    return df