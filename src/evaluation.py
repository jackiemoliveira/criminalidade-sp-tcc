from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import statsmodels.api as sm

def evaluate_models(rf, xgb_model, ols, X_train, X_test, y_train, y_test):

    results = []

    # =====================
    # OLS
    # =====================

    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    pred_train = ols.predict(X_train_const)
    pred_test = ols.predict(X_test_const)

    results.append({
        "modelo": "Regressão Linear",
        "dataset": "treino",
        "R2": r2_score(y_train, pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_train, pred_train))
    })

    results.append({
        "modelo": "Regressão Linear",
        "dataset": "teste",
        "R2": r2_score(y_test, pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_test))
    })

    # =====================
    # RANDOM FOREST
    # =====================

    pred_train = rf.predict(X_train)
    pred_test = rf.predict(X_test)

    results.append({
        "modelo": "Random Forest",
        "dataset": "treino",
        "R2": r2_score(y_train, pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_train, pred_train))
    })

    results.append({
        "modelo": "Random Forest",
        "dataset": "teste",
        "R2": r2_score(y_test, pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_test))
    })

    # =====================
    # XGBOOST
    # =====================

    pred_train = xgb_model.predict(X_train)
    pred_test = xgb_model.predict(X_test)

    results.append({
        "modelo": "XGBoost",
        "dataset": "treino",
        "R2": r2_score(y_train, pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_train, pred_train))
    })

    results.append({
        "modelo": "XGBoost",
        "dataset": "teste",
        "R2": r2_score(y_test, pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_test))
    })

    return pd.DataFrame(results)


def create_results_table(results_df, output_path):

    import pandas as pd

    tabela = results_df.pivot_table(
        index=["modelo", "cenario", "target"],
        columns="dataset",
        values=["R2", "RMSE"]
    )

    tabela = tabela.reset_index()

    tabela.columns = [
        "modelo",
        "cenario",
        "target",
        "R2_teste",
        "R2_treino",
        "RMSE_teste",
        "RMSE_treino"
    ]

    tabela.to_excel(output_path, index=False)

    return tabela