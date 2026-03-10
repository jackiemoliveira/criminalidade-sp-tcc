import pandas as pd


def compare_original_vs_treated(df_original, df_tratado, variaveis):

    comparacao = pd.DataFrame({

        "missing_original": df_original[variaveis].isnull().sum(),

        "missing_tratado": df_tratado[variaveis].isnull().sum(),

        "media_original": df_original[variaveis].mean(),

        "media_tratado": df_tratado[variaveis].mean()

    })

    comparacao = comparacao.round(3)

    return comparacao