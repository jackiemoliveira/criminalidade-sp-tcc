import pandas as pd


def load_data(path):

    df = pd.read_csv(
        path,
        delimiter=";",
        decimal=","
    )

    # converter colunas numéricas
    numeric_cols = [
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

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df

def clean_data(df):

    df_original = df.copy()

    # homicidios - substituir nulos por zero | significa ausência de registro de homícidios no período
    df["homicide_count"] = df["homicide_count"].fillna(0)

    ##mortalidade infantil - substituir nulos por imputação pela mediana | motivos robusta a outliers, mantém todos municípios e comum em estudos socioeconômicos
    df["infant_mortality"] = df["infant_mortality"].fillna(
        df["infant_mortality"].median()
    )

    ##taxa de desemprego - se o desemprego calculado for negativo: unemployment=max(0,unemployment)
    df["per_cent_unemployed"] = df["per_cent_unemployed"].clip(lower=0)
    
    return df_original, df