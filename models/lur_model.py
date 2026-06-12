"""
LUR modelling pipeline. The pollutant column in the CSV is either NO2_ugm3
or PM2_5_ugm3 (dot replaced by underscore). Both names are handled throughout.
Features are selected dynamically from whatever is non-null in the CSV.
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT      = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "eea_finland_annual.csv"
RANDOM_STATE = 42

# PM2.5 is stored with underscore in CSV, accept both forms from caller
POLLUTANT_ALIASES = {
    "PM2.5_ugm3": "PM2_5_ugm3",
    "PM2_5_ugm3": "PM2_5_ugm3",
    "NO2_ugm3":   "NO2_ugm3",
}

CANDIDATE_FEATURES = ["latitude", "longitude", "road_count_500m", "year"]


def _resolve_pollutant(name):
    return POLLUTANT_ALIASES.get(name, name)


def load_data(pollutant="NO2_ugm3"):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {DATA_PATH}. "
            "Please run: python scripts/download_eea_data.py"
        )

    df     = pd.read_csv(DATA_PATH)
    target = _resolve_pollutant(pollutant)

    if target not in df.columns:
        conc_cols = [c for c in df.columns if "ugm3" in c.lower()]
        raise ValueError(f"Column '{target}' not found. Concentration columns: {conc_cols}")

    # Select features that exist and have at least one non-null value
    features = []
    for f in CANDIDATE_FEATURES:
        if f not in df.columns:
            continue
        df[f] = pd.to_numeric(df[f], errors="coerce")
        if df[f].notna().sum() > 0:
            features.append(f)

    if not features:
        raise ValueError("No usable feature columns found.")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df_clean   = df.dropna(subset=[target] + features).reset_index(drop=True)

    if len(df_clean) == 0:
        null_counts = df[[target] + features].isnull().sum()
        raise ValueError(
            f"All rows dropped after removing NaN values.\n"
            f"Null counts:\n{null_counts.to_string()}\n"
            f"Delete the CSV and re-run the download script to rebuild it."
        )

    return df_clean, df_clean[features].values, df_clean[target].values, features


def build_models():
    return {
        "Ridge LUR": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=4,
            min_samples_leaf=3, random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=2, subsample=0.8, random_state=RANDOM_STATE,
        ),
    }


def evaluate_kfold(model, X, y, k=5):
    n_splits = min(k, len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmse_list, r2_list = [], []
    for tr, va in kf.split(X):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        rmse_list.append(np.sqrt(mean_squared_error(y[va], pred)))
        r2_list.append(r2_score(y[va], pred))
    return {
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std":  float(np.std(rmse_list)),
        "r2_mean":   float(np.mean(r2_list)),
        "r2_std":    float(np.std(r2_list)),
    }


def evaluate_loocv(model, X, y):
    yt, yp = [], []
    for tr, va in LeaveOneOut().split(X):
        model.fit(X[tr], y[tr])
        yt.append(y[va][0])
        yp.append(model.predict(X[va])[0])
    yt, yp = np.array(yt), np.array(yp)
    return {
        "loocv_rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "loocv_r2":   float(r2_score(yt, yp)),
        "y_true": yt, "y_pred": yp,
    }


def run_full_evaluation(pollutant="NO2_ugm3"):
    df, X, y, features = load_data(pollutant)
    Xs = StandardScaler().fit_transform(X)
    rows, preds = [], {}
    for name, model in build_models().items():
        kf  = evaluate_kfold(model, Xs, y)
        loo = evaluate_loocv(model, Xs, y)
        preds[name] = (loo["y_true"], loo["y_pred"])
        rows.append({
            "Model":           name,
            "5-fold RMSE":     round(kf["rmse_mean"],   2),
            "5-fold RMSE std": round(kf["rmse_std"],    2),
            "5-fold R2":       round(kf["r2_mean"],     3),
            "5-fold R2 std":   round(kf["r2_std"],      3),
            "LOOCV RMSE":      round(loo["loocv_rmse"], 2),
            "LOOCV R2":        round(loo["loocv_r2"],   3),
        })
    return pd.DataFrame(rows), preds


def get_ridge_coefficients(pollutant="NO2_ugm3"):
    df, X, y, features = load_data(pollutant)
    Xs = StandardScaler().fit_transform(X)
    m  = Ridge(alpha=1.0)
    m.fit(Xs, y)
    return pd.DataFrame({
        "Feature":         features,
        "Coefficient":     m.coef_,
        "Abs Coefficient": np.abs(m.coef_),
    }).sort_values("Abs Coefficient", ascending=False).reset_index(drop=True)


def get_gb_importance(pollutant="NO2_ugm3"):
    df, X, y, features = load_data(pollutant)
    Xs = StandardScaler().fit_transform(X)
    m  = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=2, subsample=0.8, random_state=RANDOM_STATE
    )
    m.fit(Xs, y)
    return pd.DataFrame({
        "Feature":    features,
        "Importance": m.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)


def get_dataset_summary():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    return df


def get_active_features(pollutant="NO2_ugm3"):
    try:
        _, _, _, features = load_data(pollutant)
        return features
    except Exception:
        return []
