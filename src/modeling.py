from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.config import TEST_SIZE, RANDOM_STATE
import statsmodels.api as sm



def train_models(df, features, target):

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42
    )

    rf.fit(X_train, y_train)

    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    # OLS
    X_train_const = sm.add_constant(X_train)

    ols = sm.OLS(
        y_train,
        X_train_const
    ).fit()

    return rf, xgb_model, ols, X_train, X_test, y_train, y_test