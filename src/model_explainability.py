import shap
import matplotlib.pyplot as plt


def shap_analysis(model, X, output_path):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.title(
    "Impacto das variáveis na predição das taxas de homicídio municipais (XGBoost – SHAP)",
    fontsize=12
    )  

    plt.xlabel(
    "Valor SHAP (impacto na taxa prevista de homicídios – escala log)"
    )

    plt.tight_layout()
    
    plt.savefig(output_path)

    plt.close()