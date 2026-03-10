import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_pca_analysis(df, features, output_path):

    data = df[features].dropna()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=2)

    components = pca.fit_transform(X_scaled)

    pca_df = data.copy()

    pca_df["PC1"] = components[:, 0]
    pca_df["PC2"] = components[:, 1]

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        x="PC1",
        y="PC2",
        data=pca_df
    )

    plt.title("PCA dos indicadores socioeconômicos")

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variância)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variância)")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

    return pca