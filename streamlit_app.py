"""\
Minimal interactive dashboard for the Phase 3 final defense.

Goal: present the "Engine" (model) + a simple "Dashboard" view.
- Uses ONLY pre-launch features.
- Lets you adjust decision threshold to show recall/precision trade-off.

Run:
  streamlit run streamlit_app.py

Notes:
- Keeps the feature engineering consistent with the notebook/scripts.
- Intended to be minimal (single page, no extra bells and whistles).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data/steam.csv"


def parse_owners(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(",", "").replace("–", "-").replace("—", "-")
    m = re.match(r"^(\d+(?:\.\d+)?)(k)?\s*-\s*(\d+(?:\.\d+)?)(k)?$", s)
    if m:
        lo = float(m.group(1)) * (1000.0 if m.group(2) else 1.0)
        hi = float(m.group(3)) * (1000.0 if m.group(4) else 1.0)
        return (lo + hi) / 2.0
    m2 = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m2:
        return (float(m2.group(1)) + float(m2.group(2))) / 2.0
    return np.nan


def to01(x):
    if pd.isna(x):
        return np.nan
    try:
        return 1 if int(float(x)) != 0 else 0
    except Exception:
        s = str(x).strip().lower()
        if s in {"true", "t", "yes", "y"}:
            return 1
        if s in {"false", "f", "no", "n"}:
            return 0
        return np.nan


def normstr(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    return s if s else None


def get_season(month):
    if pd.isna(month):
        return None
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Fall"


FEATURE_COLS = [
    "english",
    "has_multiplayer",
    "has_publisher",
    "platform_count",
    "name_length",
    "is_mature",
    "is_free",
    "season_Spring",
    "season_Summer",
    "season_Fall",
    "season_Winter",
]


@st.cache_data(show_spinner=False)
def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, on_bad_lines="skip")


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df = df.copy()

    df["owners_numeric"] = df["owners"].map(parse_owners)
    median = float(df["owners_numeric"].median())
    df["success"] = (df["owners_numeric"] > median).astype(int)

    df["english"] = df["english"].map(to01)
    df["has_multiplayer"] = df["categories"].fillna("").str.contains(
        "multi-player|multiplayer|co-op|coop", case=False, regex=True
    ).astype(int)

    dev = df["developer"].map(normstr)
    pub = df["publisher"].map(normstr)
    df["has_publisher"] = ((dev.notna()) & (pub.notna()) & (dev != pub)).astype(int)

    rel = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_month"] = rel.dt.month
    df["season"] = df["release_month"].apply(get_season)
    season_dummies = pd.get_dummies(df["season"], prefix="season")
    df = pd.concat([df, season_dummies], axis=1)

    if "platforms" in df.columns:
        df["platform_count"] = df["platforms"].fillna("").str.split(";").apply(len)
        df.loc[df["platforms"].isna(), "platform_count"] = 0
    else:
        df["platform_count"] = 1

    df["name_length"] = df["name"].fillna("").str.len()

    if "price" in df.columns:
        df["price_clean"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price_clean"] = 0

    if "required_age" in df.columns:
        df["is_mature"] = (pd.to_numeric(df["required_age"], errors="coerce") >= 17).astype(int)
    else:
        df["is_mature"] = 0

    df["is_free"] = (df["price_clean"] == 0).astype(int)

    # Ensure all season columns exist
    for col in ["season_Spring", "season_Summer", "season_Fall", "season_Winter"]:
        if col not in df.columns:
            df[col] = 0

    df_clean = df.dropna(subset=FEATURE_COLS + ["success"]).copy()
    return df_clean, median


def build_model(model_name: str, y_train: pd.Series):
    if model_name == "Logistic Regression (balanced)":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    if model_name == "Random Forest (balanced)":
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        )

    if model_name == "XGBoost (recall-first)":
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost not installed. Install xgboost or choose another model.")
        scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        )

    raise ValueError(f"Unknown model: {model_name}")


def get_feature_importances(model, feature_names: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        vals = getattr(model, "feature_importances_")
        return (
            pd.DataFrame({"feature": feature_names, "importance": vals})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    if hasattr(model, "coef_"):
        coef = model.coef_.ravel()
        return (
            pd.DataFrame({"feature": feature_names, "importance": np.abs(coef), "signed": coef})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return None


def main():
    st.set_page_config(page_title="Steam Visibility Dashboard", layout="wide")

    st.title("Steam Games — Pre-Launch Visibility Dashboard")
    st.caption("Phase 3: Engine + Dashboard (minimal, recall-first)")

    with st.sidebar:
        st.header("Ayarlar")
        dataset_path = st.text_input("Dataset yolu", value=str(DEFAULT_DATASET))

        model_choices = [
            "Logistic Regression (balanced)",
            "Random Forest (balanced)",
        ]
        if HAS_XGBOOST:
            model_choices.insert(0, "XGBoost (recall-first)")

        model_name = st.selectbox("Model", options=model_choices, index=0)
        threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

        st.divider()
        st.write("**FN maliyeti** ve **FP maliyeti** (temsili):")
        fn_cost = st.number_input("FN cost ($)", min_value=0, value=10000, step=1000)
        fp_cost = st.number_input("FP cost ($)", min_value=0, value=1000, step=100)

    if not Path(dataset_path).exists():
        st.error(f"Dataset bulunamadı: {dataset_path}")
        st.stop()

    raw = load_raw_dataset(dataset_path)
    df, owners_median = engineer_features(raw)

    X = df[FEATURE_COLS].astype(float)
    y = df["success"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(model_name, y_train)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: treat predict as hard label
        y_proba = model.predict(X_test).astype(float)

    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = fn * fn_cost + fp * fp_cost

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1", f"{f1:.3f}")
    col5.metric("Cost (FN/FP)", f"${total_cost:,.0f}")

    st.write(
        f"**Target tanımı:** owners_numeric > median (median = {owners_median:,.0f}).  "
        f"**Test set:** {len(X_test):,} örnek | **Positives (hits):** {(y_test==1).sum():,}"
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Confusion Matrix")
        st.write(f"TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            square=True,
            linewidths=1.5,
            linecolor="black",
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Low", "High"])
        ax.set_yticklabels(["Low", "High"])
        st.pyplot(fig)

    with right:
        st.subheader("Precision–Recall Curve")
        precs, recs, ths = precision_recall_curve(y_test, y_proba)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(recs, precs)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title("PR Curve (threshold slider affects CM)")
        st.pyplot(fig)

    st.subheader("Feature Importance / Coefficients")
    imp = get_feature_importances(model, FEATURE_COLS)
    if imp is None:
        st.info("Bu model için feature importance alınamadı.")
    else:
        topk = imp.head(12)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(topk["feature"][::-1], topk["importance"][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Top Features")
        ax.grid(True, axis="x", alpha=0.3)
        st.pyplot(fig)

    st.subheader("Actionable Game/Business Insights (özet)")
    st.write(
        "- Publisher backing ve multiplayer sinyali başarıyla pozitif ilişkili görünüyor.\n"
        "- English desteği global görünürlüğü artırıyor.\n"
        "- Season etkisi düşük; timing yerine ürün/dağıtım sinyalleri daha önemli.\n"
        "- Threshold ile: recall ↑ yapınca FP ↑; bütçeye göre dengele."
    )


if __name__ == "__main__":
    main()
