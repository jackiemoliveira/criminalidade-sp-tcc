import numpy as np


def calculate_bayes_rate(df):

    df = df.copy()

    df["tx_homicidios_observada"] = (
        df["homicide_count"] / df["resident_population"]
    ) * 100000

    taxa_media = df["homicide_count"].sum() / df["resident_population"].sum()

    df["tx_homicidios_bayes"] = (
        (df["homicide_count"] + taxa_media * df["resident_population"])
        /
        (df["resident_population"] + df["resident_population"].mean())
    ) * 100000

    return df