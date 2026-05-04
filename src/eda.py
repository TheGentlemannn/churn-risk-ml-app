import os

from config import REPORTS_DIR

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import TARGET
from data_loader import clean_data, load_data


sns.set_theme(style="whitegrid", palette="Set2")


def save_churn_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=TARGET)
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customers")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "churn_distribution.png", dpi=160)
    plt.close()


def save_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax)
        ax.set_title(f"{col} by Churn")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "numeric_distributions.png", dpi=160)
    plt.close()


def save_contract_churn(df: pd.DataFrame) -> None:
    contract_rate = (
        df.assign(churn_flag=df[TARGET].map({"No": 0, "Yes": 1}))
        .groupby("Contract", as_index=False)["churn_flag"]
        .mean()
        .sort_values("churn_flag", ascending=False)
    )

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=contract_rate, x="Contract", y="churn_flag")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Churn Rate")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "contract_churn_rate.png", dpi=160)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    encoded = df.copy()
    encoded[TARGET] = encoded[TARGET].map({"No": 0, "Yes": 1})
    corr = encoded[["tenure", "MonthlyCharges", "TotalCharges", TARGET]].corr()

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f")
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def run_eda() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_data(load_data())

    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(10))
    print("\nChurn rate:\n", df[TARGET].value_counts(normalize=True))
    print("\nNumeric summary:\n", df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())

    save_churn_distribution(df)
    save_numeric_distributions(df)
    save_contract_churn(df)
    save_correlation_heatmap(df)
    print(f"EDA charts saved to {REPORTS_DIR}")


if __name__ == "__main__":
    run_eda()
