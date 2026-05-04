import pandas as pd

from config import DATA_PATH, DATA_URL


def download_data(force: bool = False) -> pd.DataFrame:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        return pd.read_csv(DATA_PATH)

    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)
    return df


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return download_data()
    return pd.read_csv(DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])

    return df


if __name__ == "__main__":
    data = download_data(force=True)
    print(f"Downloaded {data.shape[0]} rows and {data.shape[1]} columns to {DATA_PATH}")
