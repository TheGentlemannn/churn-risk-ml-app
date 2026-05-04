import json
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import (
    FEATURE_IMPORTANCE_PATH,
    ID_COLUMNS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    SHAP_PLOT_PATH,
    TARGET,
    TEST_SIZE,
)
from data_loader import clean_data, load_data

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / ".matplotlib"))

import matplotlib.pyplot as plt


def split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET].map({"No": 0, "Yes": 1})
    X = df.drop(columns=[TARGET] + [col for col in ID_COLUMNS if col in df.columns])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def get_models() -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    return preprocessor.get_feature_names_out()


def save_feature_importance(best_model: Pipeline) -> None:
    classifier = best_model.named_steps["model"]
    feature_names = get_feature_names(best_model.named_steps["preprocessor"])

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(25)
    )
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    plt.figure(figsize=(9, 7))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.title("Top 25 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=160)
    plt.close()


def save_shap_summary(best_model: Pipeline, X_sample: pd.DataFrame) -> None:
    try:
        import shap

        preprocessor = best_model.named_steps["preprocessor"]
        classifier = best_model.named_steps["model"]
        transformed = preprocessor.transform(X_sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        feature_names = get_feature_names(preprocessor)
        transformed_df = pd.DataFrame(transformed, columns=feature_names)

        if hasattr(classifier, "feature_importances_"):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_df)
            shap.summary_plot(shap_values, transformed_df, show=False, max_display=20)
        else:
            explainer = shap.LinearExplainer(classifier, transformed_df)
            shap_values = explainer(transformed_df)
            shap.plots.beeswarm(shap_values, show=False, max_display=20)

        plt.tight_layout()
        plt.savefig(SHAP_PLOT_PATH, dpi=160, bbox_inches="tight")
        plt.close()
        error_path = REPORTS_DIR / "shap_error.json"
        if error_path.exists():
            error_path.unlink()
    except Exception as exc:
        with open(REPORTS_DIR / "shap_error.json", "w", encoding="utf-8") as file:
            json.dump({"error": str(exc)}, file, indent=2)


def train() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_data())
    X, y = split_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []
    trained_models = {}

    for name, model in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        results.append(evaluate_model(name, pipeline, X_test, y_test))
        trained_models[name] = pipeline

    metrics = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    metrics.to_csv(METRICS_PATH, index=False)

    best_name = metrics.iloc[0]["model"]
    best_model = trained_models[best_name]
    joblib.dump(best_model, MODEL_PATH)

    save_feature_importance(best_model)
    save_shap_summary(best_model, X_test.sample(min(300, len(X_test)), random_state=RANDOM_STATE))

    print(metrics.round(4).to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
