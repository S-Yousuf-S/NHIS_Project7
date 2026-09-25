"""
================================================================================
 PREREQUISITES.PY

 Project      : Disaster Tweet Classification (NLP, Binary Classification)

 Author       : Yousuf S. R. Sakkaf

 Description  : Shared, project-agnostic utility module imported by the
                single project notebook (supervisor mandate, 24 Sep 2026):
                disaster_tweet_classification.ipynb
                All tiers — classical ML, BiLSTM, and DistilBERT (mandatory,
                supervisor's requested model) — live in this one notebook.

 Scope        : Only contains logic that is genuinely reusable across ANY
                project/dataset - no dataset-specific decisions (e.g. which
                strategy to use on which column, which features to engineer,
                which keyword-extraction technique). Those decisions are made
                explicitly and visibly inside the notebook, not hidden in a
                function here.

 Contents     : - Data loading cascade: Local -> Google Drive -> GitHub raw
                - Model artifact save/load cascade (check-before-retrain,
                  matches Project 6's pattern) - critical here since a single
                  notebook now carries a mandatory fine-tuned DistilBERT tier
                - Logging configuration
                - HTML theme, color palette & plot style (matplotlib/seaborn)
                - Generic audit functions (missing values, duplicates - report only)
                - Generic statistical formulas (IQR, Z-score, skew, kurtosis)
                - Generic text-structure counters (URL/mention/hashtag counts) -
                  reusable across any text dataset, not disaster-tweet-specific

 Usage        : from prerequisites import *

 Note         : Color theme below is a placeholder pending the deck's final
                Twitter-style light/sky-blue direction (TBD) - swap
                PRIMARY_COLOR/SECONDARY_COLOR etc. once that's locked,
                everything else here is theme-agnostic.
================================================================================
"""

import os
import re
import glob
import logging
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, zscore
from IPython.display import display, HTML

# =========================================================
# COLOR THEME (placeholder — confirm once deck theme is locked)
# =========================================================
PRIMARY_COLOR = "indigo"
SECONDARY_COLOR = "thistle"
SUCCESS_COLOR = "#2E8B57"
WARNING_COLOR = "#C0392B"
INFO_COLOR = "#1F77B4"
TABLE_COLOR = "mediumpurple"
TITLE_COLOR = "midnightblue"


# =========================================================
# LOGGING
# =========================================================
def setup_logging(log_dir="logs", name="Project7"):
    """
    Sets up a logger writing to a date-only filename (YS_ prefix), append
    mode, DEBUG level, with both file and console handlers. Logger name
    is included in the formatter for cross-notebook differentiation.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"YS_{datetime.now().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_filename, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized -> {log_filename}")
    return logger


# =========================================================
# HTML STATUS BOXES / SECTION HEADERS
# =========================================================
def section_header(title):
    display(HTML(f"""
    <div style="background-color:{PRIMARY_COLOR}; color:white; padding:12px 18px;
                border-radius:6px; margin-top:24px; margin-bottom:12px;">
        <h2 style="margin:0;">{title}</h2>
    </div>
    """))


def subsection_header(title):
    display(HTML(f"""
    <div style="border-left:5px solid {SECONDARY_COLOR}; padding:6px 14px;
                margin-top:16px; margin-bottom:8px;">
        <h3 style="margin:0; color:{TITLE_COLOR};">{title}</h3>
    </div>
    """))


def success_box(message):
    display(HTML(f"""
    <div style="background-color:#EAF7EF; border-left:5px solid {SUCCESS_COLOR};
                padding:10px 16px; margin:8px 0; border-radius:4px;">
        <b style="color:{SUCCESS_COLOR};">Success:</b> {message}
    </div>
    """))


def warning_box(message):
    display(HTML(f"""
    <div style="background-color:#FBEAEA; border-left:5px solid {WARNING_COLOR};
                padding:10px 16px; margin:8px 0; border-radius:4px;">
        <b style="color:{WARNING_COLOR};">Warning:</b> {message}
    </div>
    """))


def info_box(message):
    display(HTML(f"""
    <div style="background-color:#EAF2FB; border-left:5px solid {INFO_COLOR};
                padding:10px 16px; margin:8px 0; border-radius:4px;">
        <b style="color:{INFO_COLOR};">Info:</b> {message}
    </div>
    """))


def centered_table(df, max_rows=20):
    """Renders a DataFrame as a centered, styled HTML table."""
    html = df.head(max_rows).to_html(classes="styled-table", border=0)
    display(HTML(f"""
    <div style="display:flex; justify-content:center; margin:12px 0;">
        <style>
            .styled-table {{ border-collapse: collapse; font-size:0.9em; }}
            .styled-table th {{ background-color:{TABLE_COLOR}; color:white; padding:6px 12px; }}
            .styled-table td {{ padding:6px 12px; border-bottom:1px solid #ddd; }}
        </style>
        {html}
    </div>
    """))


# =========================================================
# VISUALIZATION DEFAULTS
# =========================================================
def set_global_visualization_settings(palette="viridis"):
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 1000)
    sns.set_theme(style="whitegrid", palette=palette)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


# =========================================================
# DATA LOADING CASCADE (Local -> Drive -> GitHub)
# =========================================================
def load_data_file(
    filename,
    local_dir="Assets",
    drive_path=None,  # CONFIRM exact Drive path before relying on this fallback
    github_raw_base="https://raw.githubusercontent.com/S-Yousuf-S/NHIS_Project7/main/Assets",
    dataset_name="dataset",
):
    """
    Attempts to load `filename` via: Local -> Google Drive -> GitHub raw URL.
    drive_path is a placeholder - confirm the exact mounted Drive folder path
    before this fallback is exercised (a guessed path cost time on a prior
    project - verify first).
    """
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        logging.getLogger("Project7").info(f"[{dataset_name}] Loaded locally: {local_path}")
        return pd.read_csv(local_path)

    if drive_path:
        drive_file = os.path.join(drive_path, filename)
        if os.path.exists(drive_file):
            logging.getLogger("Project7").info(f"[{dataset_name}] Loaded from Drive: {drive_file}")
            return pd.read_csv(drive_file)

    github_url = f"{github_raw_base}/{filename}"
    logging.getLogger("Project7").info(f"[{dataset_name}] Falling back to GitHub: {github_url}")
    return pd.read_csv(github_url)


# =========================================================
# MODEL ARTIFACT SAVE/LOAD CASCADE (check-before-retrain)
#
# Call save_model_artifact() immediately after a model finishes training -
# not at notebook end - so a Colab disconnect only costs you whatever hasn't
# been serialized yet. Call load_latest_model_artifact() at the START of
# each model's training cell: if it returns a model, skip training entirely
# and reuse it - this is what protects the mandatory fine-tuned DistilBERT
# tier (and everything else) from a 30+ minute retrain on every rerun.
# =========================================================
def save_model_artifact(model, model_name, local_dir="Assets/models", framework="joblib"):
    """
    Saves a trained model with a timestamp in its filename.
    framework: "joblib" (sklearn/xgboost/lightgbm), "keras" (BiLSTM),
               "torch" (DistilBERT / HF Trainer checkpoints - pass the
               model's .save_pretrained-compatible object or state_dict
               path handling as needed inside the notebook; this function
               covers the joblib/keras cases directly).
    """
    os.makedirs(local_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if framework == "keras":
        filename = f"{model_name}_{timestamp}.keras"
        model.save(os.path.join(local_dir, filename))
    else:
        filename = f"{model_name}_{timestamp}.pkl"
        joblib.dump(model, os.path.join(local_dir, filename))

    logging.getLogger("Project7").info(f"Saved model artifact: {filename}")
    return filename


def load_latest_model_artifact(model_name, local_dir="Assets/models", drive_dir=None, framework="joblib"):
    """
    Checks for an existing serialized model before retraining, cascading
    Local -> Drive (GitHub artifact fallback intentionally omitted - binary
    model files are a poor fit for a git repo; Drive is the durable store
    for these). Returns None if nothing is found, meaning the caller should
    train from scratch.
    drive_dir is a placeholder - confirm the exact mounted Drive path before
    relying on this fallback, same caution as load_data_file().
    """
    ext = "keras" if framework == "keras" else "pkl"

    def _latest_match(directory):
        pattern = os.path.join(directory, f"{model_name}_*.{ext}")
        matches = sorted(glob.glob(pattern))
        return matches[-1] if matches else None

    latest = _latest_match(local_dir)
    source = "local"
    if not latest and drive_dir:
        latest = _latest_match(drive_dir)
        source = "Drive"

    if not latest:
        logging.getLogger("Project7").info(f"No existing artifact for {model_name} - training required")
        return None

    logging.getLogger("Project7").info(f"Found existing {source} artifact, skipping retraining: {latest}")
    if framework == "keras":
        from tensorflow import keras
        return keras.models.load_model(latest)
    return joblib.load(latest)


# =========================================================
# GENERIC DIAGNOSTICS (dataset-agnostic — usable on any numeric feature,
# e.g. tweet length, hashtag/mention counts, keyword-presence flags)
# =========================================================
def missing_value_audit(df, dataset_name="dataset"):
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    audit = pd.DataFrame({"missing_count": missing, "missing_pct": pct.round(2)})
    audit = audit[audit["missing_count"] > 0].sort_values("missing_count", ascending=False)
    logging.getLogger("Project7").info(f"[{dataset_name}] Missing-value audit: {len(audit)} columns affected")
    return audit


def duplicate_check(df, subset=None, dataset_name="dataset"):
    dupe_count = df.duplicated(subset=subset).sum()
    logging.getLogger("Project7").info(f"[{dataset_name}] Duplicate rows found: {dupe_count}")
    return dupe_count


def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def detect_outliers_iqr(series, k=1.5):
    lower, upper = iqr_bounds(series, k)
    return series[(series < lower) | (series > upper)]


def detect_outliers_zscore(series, threshold=3):
    z = zscore(series.dropna())
    return series.dropna()[np.abs(z) > threshold]


def distribution_summary(series, sample_size=None, random_state=42):
    data = series.dropna()
    if sample_size and len(data) > sample_size:
        data = data.sample(sample_size, random_state=random_state)
    return {
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "skew": skew(data),
        "kurtosis": kurtosis(data),
    }


# =========================================================
# TEXT-SPECIFIC HELPERS (new for this project — no Project 6 equivalent)
# =========================================================
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")


def count_urls(text):
    return len(URL_PATTERN.findall(str(text)))


def count_mentions(text):
    return len(MENTION_PATTERN.findall(str(text)))


def count_hashtags(text):
    return len(HASHTAG_PATTERN.findall(str(text)))
