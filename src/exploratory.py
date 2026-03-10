from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def descriptive_statistics(df, variaveis):

    tabela = df[variaveis].describe().T

    tabela = tabela[["mean", "std", "min", "max"]]

    return tabela.round(3)