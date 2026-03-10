import geopandas as gpd
import matplotlib.pyplot as plt

from libpysal.weights import Queen
from esda.moran import Moran


def calculate_morans_i(df, geo_path, output_path):

    gdf = gpd.read_file(geo_path)

    print("Município ilha (sem vizinhos):")
    print(gdf.iloc[232])

    gdf["codigo"] = gdf["id"].astype(str)

    gdf = gdf.merge(
        df,
        left_on="codigo",
        right_on="codigo",
        how="left"
    )

    gdf = gdf.dropna(subset=["tx_homicidios_bayes"])

    w = Queen.from_dataframe(gdf, use_index=False)

    w.transform = "r"

    y = gdf["tx_homicidios_bayes"].values

    moran = Moran(y, w)

    plt.figure(figsize=(6,6))

    plt.scatter(
        y,
        w.sparse.dot(y)
    )

    plt.xlabel("Taxa de homicídios")
    plt.ylabel("Lag espacial")

    plt.title(
        f"Moran's I = {moran.I:.3f} (p={moran.p_sim:.4f})"
    )

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

    return moran