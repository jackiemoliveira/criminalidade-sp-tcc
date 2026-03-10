import pandas as pd


def data_quality_report(df):

    report = pd.DataFrame()

    report["variavel"] = df.columns
    report["tipo_dado"] = df.dtypes.values
    report["missing"] = df.isnull().sum().values
    report["missing_percent"] = (df.isnull().sum() / len(df) * 100).round(2).values
    report["valores_unicos"] = df.nunique().values

    # estatísticas numéricas
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    stats = df[numeric_cols].describe().T

    stats = stats[["mean", "std", "min", "max"]].round(3)

    report = report.merge(
        stats,
        left_on="variavel",
        right_index=True,
        how="left"
    )

    return report