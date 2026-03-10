def generate_scenarios(df):

    df = df.copy()

    df["log_tx_homicidios_bayes"] = np.log1p(df["tx_homicidios_bayes"])

    df["log_populacao"] = np.log1p(df["resident_population"])

    df["log_pib"] = np.log1p(df["gdp_per_capita"])

    return {"modelo_log": df}