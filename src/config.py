from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

DATA_URL = (
    "https://raw.githubusercontent.com/treselle-systems/"
    "customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
DATA_PATH = DATA_DIR / "telco_churn.csv"
MODEL_PATH = MODEL_DIR / "best_churn_model.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
SHAP_PLOT_PATH = REPORTS_DIR / "shap_summary.png"

TARGET = "Churn"
ID_COLUMNS = ["customerID"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
