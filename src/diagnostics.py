import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(df, features, output_path):
    """
    Calcula o Variance Inflation Factor (VIF) para avaliar
    multicolinearidade entre variáveis explicativas.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contendo as variáveis
    features : list
        Lista de variáveis explicativas
    output_path : str
        Caminho para salvar o resultado

    Returns
    -------
    pandas.DataFrame
        Tabela com valores de VIF
    """

    X = df[features].copy()
    X_const = sm.add_constant(X)

    vif_data = pd.DataFrame()

    vif_data["variavel"] = features
    vif_data["VIF"] = [
        variance_inflation_factor(X_const.values, i + 1)
        for i in range(len(features))
    ]

    vif_data.to_csv(output_path, index=False)

    return vif_data