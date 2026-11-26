"""Streamlit interface for the Employee Salary Prediction model.

The saved Joblib file is deliberately treated as the source of truth for
categories, preprocessing column order, and the fitted scikit-learn objects.
No preprocessing object is fitted in this application.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_NAME = "employee_salary_model.joblib"
INPUT_FEATURES = [
    "job_title",
    "experience_years",
    "education_level",
    "skills_count",
    "industry",
    "company_size",
    "location",
    "remote_work",
    "certifications",
]
MODEL_METRICS = {
    "MAE": "5,436.10",
    "MSE": "50,773,076.35",
    "RMSE": "7,125.52",
    "R² Score": "0.9635",
}


@dataclass(frozen=True)
class ModelArtifacts:
    """The exact fitted objects and schema persisted during model training."""

    model: Any
    encoder: Any
    scaler: Any
    categorical_columns: list[str]
    numeric_columns: list[str]


def artifact_path() -> Path:
    """Support the current project layout and an optional artifacts directory."""
    candidates = (APP_DIR / ARTIFACT_NAME, APP_DIR / "artifacts" / ARTIFACT_NAME)
    for path in candidates:
        if path.exists():
            return path
    searched = " or ".join(str(path.relative_to(APP_DIR)) for path in candidates)
    raise FileNotFoundError(f"Could not find {ARTIFACT_NAME} in {searched}.")


@st.cache_resource(show_spinner="Loading the trained salary model…")
def load_artifacts(path_as_text: str) -> ModelArtifacts:
    """Load and validate the fitted training artifacts once per app session."""
    payload = joblib.load(path_as_text)
    required_keys = {"model", "encoder", "scaler", "categorical_col", "numeric_col"}
    if not isinstance(payload, dict) or not required_keys.issubset(payload):
        raise ValueError(
            "The model artifact must be a dictionary containing model, encoder, "
            "scaler, categorical_col, and numeric_col."
        )

    artifacts = ModelArtifacts(
        model=payload["model"],
        encoder=payload["encoder"],
        scaler=payload["scaler"],
        categorical_columns=list(payload["categorical_col"]),
        numeric_columns=list(payload["numeric_col"]),
    )

    expected_columns = set(INPUT_FEATURES)
    artifact_columns = set(artifacts.categorical_columns + artifacts.numeric_columns)
    if artifact_columns != expected_columns:
        raise ValueError(
            "The artifact columns do not match the nine expected salary input features. "
            f"Found: {sorted(artifact_columns)}"
        )
    if len(artifacts.encoder.categories_) != len(artifacts.categorical_columns):
        raise ValueError("The encoder category groups do not match categorical_col.")

    expected_features = len(artifacts.numeric_columns) + sum(
        len(categories) for categories in artifacts.encoder.categories_
    )
    model_features = getattr(artifacts.model, "n_features_in_", None)
    scaler_features = getattr(artifacts.scaler, "n_features_in_", None)
    if expected_features != 48:
        raise ValueError(
            f"The saved preprocessing schema produces {expected_features} features; "
            "this application requires the model's 48-feature schema."
        )
    if model_features != expected_features or scaler_features != expected_features:
        raise ValueError(
            "The model, scaler, and encoder disagree on their expected feature count."
        )
    return artifacts


def category_options(artifacts: ModelArtifacts, column: str) -> list[Any]:
    """Return only categories learned by the fitted encoder for a field."""
    index = artifacts.categorical_columns.index(column)
    return list(artifacts.encoder.categories_[index])


def build_feature_matrix(values: dict[str, Any], artifacts: ModelArtifacts) -> np.ndarray:
    """Recreate the training feature matrix: numeric values then one-hot columns.

    The model artifact shows that StandardScaler was fitted after concatenating
    these 3 numeric and 45 one-hot columns. This function never calls fit.
    """
    row = pd.DataFrame([values], columns=INPUT_FEATURES)
    numeric_values = row[artifacts.numeric_columns].to_numpy(dtype=float)
    categorical_values = row[artifacts.categorical_columns]

    # The supplied encoder was trained with handle_unknown="ignore". Retain that
    # safety behavior if this helper is later called outside the dropdown form.
    learned_categories = {
        column: set(map(str, category_options(artifacts, column)))
        for column in artifacts.categorical_columns
    }
    unknown_values = [
        column
        for column in artifacts.categorical_columns
        if str(values[column]) not in learned_categories[column]
    ]
    if unknown_values and getattr(artifacts.encoder, "handle_unknown", None) != "ignore":
        raise ValueError(
            "Unknown categorical values are not supported by this trained encoder: "
            f"{', '.join(unknown_values)}"
        )

    encoded_values = artifacts.encoder.transform(categorical_values)
    if hasattr(encoded_values, "toarray"):
        encoded_values = encoded_values.toarray()
    feature_matrix = np.hstack((numeric_values, np.asarray(encoded_values)))

    expected_features = artifacts.scaler.n_features_in_
    if feature_matrix.shape != (1, expected_features):
        raise ValueError(
            f"Expected a 1 × {expected_features} feature matrix, "
            f"but created {feature_matrix.shape}."
        )
    return feature_matrix


def predict_salary(values: dict[str, Any], artifacts: ModelArtifacts) -> float:
    """Run the original fitted encoder, scaler, and model in training order."""
    feature_matrix = build_feature_matrix(values, artifacts)
    scaled_features = artifacts.scaler.transform(feature_matrix)
    prediction = float(artifacts.model.predict(scaled_features)[0])
    if not math.isfinite(prediction):
        raise ValueError("The trained model returned an invalid salary prediction.")
    return prediction


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink: #edf5ff; --muted: #9fb4d3; --line: rgba(151, 184, 231, .18);
                --navy-950: #041126; --navy-900: #071b3c; --navy-800: #0b2855;
                --blue: #4f9cff; --cyan: #76d4ff; --surface: rgba(10, 35, 75, .78);
            }
            .stApp {
                background:
                    radial-gradient(circle at 80% -12%, rgba(37, 120, 255, .25), transparent 30rem),
                    radial-gradient(circle at 7% 35%, rgba(43, 167, 255, .12), transparent 25rem),
                    linear-gradient(135deg, var(--navy-950) 0%, #071b3c 50%, #041126 100%);
                color: var(--ink);
            }
            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #061a39 0%, #041126 100%);
                border-right: 1px solid var(--line);
            }
            [data-testid="stSidebar"] > div:first-child { padding-top: 1.15rem; }
            [data-testid="stSidebar"] .stRadio label {
                border-radius: 10px; margin: .12rem 0; padding: .5rem .55rem;
                transition: background .18s ease;
            }
            [data-testid="stSidebar"] .stRadio label:hover { background: rgba(91, 158, 255, .12); }
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #9fb4d3; }
            .block-container { max-width: 1210px; padding-top: 2.2rem; padding-bottom: 3.5rem; }
            h1, h2, h3 { color: var(--ink); letter-spacing: -.035em; }
            p, .stMarkdown { color: #c8d8ef; }
            .eyebrow {
                color: var(--cyan); font-size: .72rem; font-weight: 800; letter-spacing: .14em;
                text-transform: uppercase; margin-bottom: .8rem;
            }
            .hero {
                position: relative; overflow: hidden; padding: 2.85rem 2.8rem; border: 1px solid var(--line);
                border-radius: 24px; background: linear-gradient(120deg, rgba(16, 60, 121, .92), rgba(8, 28, 62, .84));
                box-shadow: 0 24px 56px rgba(0, 0, 0, .28); margin-bottom: 1.55rem;
            }
            .hero::after {
                content: ""; position: absolute; width: 18rem; height: 18rem; right: -7rem; top: -10rem;
                border: 1px solid rgba(135, 202, 255, .25); border-radius: 50%; box-shadow: 0 0 0 2.6rem rgba(91, 156, 255, .045), 0 0 0 5.2rem rgba(91, 156, 255, .03);
            }
            .hero h1 { max-width: 44rem; margin: 0 0 .65rem; font-size: clamp(2.2rem, 4vw, 3.75rem); line-height: 1.05; }
            .hero p { max-width: 37rem; margin: 0; font-size: 1.05rem; line-height: 1.65; color: #cfe1ff; }
            .hero-badge {
                display: inline-flex; align-items: center; gap: .45rem; margin-top: 1.35rem; padding: .42rem .7rem;
                color: #dbeaff; font-size: .79rem; font-weight: 650; border: 1px solid rgba(150, 204, 255, .2);
                border-radius: 999px; background: rgba(3, 19, 45, .28);
            }
            .hero-badge::before { content: ""; width: .45rem; height: .45rem; border-radius: 50%; background: #55dfaf; box-shadow: 0 0 0 .22rem rgba(85, 223, 175, .12); }
            .section-heading { margin: 1.7rem 0 .3rem; font-size: 1.55rem; }
            .section-lead { color: var(--muted); margin: 0 0 1.2rem; }
            .surface-card {
                height: 100%; box-sizing: border-box; padding: 1.35rem 1.4rem; border: 1px solid var(--line);
                border-radius: 18px; background: linear-gradient(145deg, rgba(17, 53, 105, .82), rgba(7, 27, 60, .82));
                box-shadow: 0 14px 32px rgba(0, 0, 0, .17);
            }
            .surface-card h3 { margin: .65rem 0 .45rem; font-size: 1.08rem; }
            .surface-card p { margin: 0; color: #b7c9e5; line-height: 1.58; }
            .card-index { color: var(--cyan); font-size: .72rem; font-weight: 800; letter-spacing: .12em; }
            .prediction-intro {
                display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin: 1.75rem 0 1rem;
                padding: 1rem 1.2rem; border: 1px solid var(--line); border-radius: 14px; background: rgba(9, 38, 82, .5);
            }
            .prediction-intro p { margin: 0; color: #b8c9e4; }
            .prediction-intro strong { display: block; color: #f1f7ff; margin-bottom: .18rem; }
            .input-count { color: var(--cyan); font-size: .77rem; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; white-space: nowrap; }
            [data-testid="stForm"] {
                background: linear-gradient(145deg, rgba(14, 48, 99, .94), rgba(7, 29, 65, .94));
                border: 1px solid var(--line); border-radius: 20px; padding: 1.55rem; box-shadow: 0 18px 38px rgba(0, 0, 0, .22);
            }
            [data-testid="stForm"] label, [data-testid="stForm"] [data-testid="stWidgetLabel"] p { color: #dce9fb !important; font-weight: 650; }
            [data-testid="stForm"] [data-baseweb="select"] > div,
            [data-testid="stForm"] input {
                background: #081f45 !important; color: #edf5ff !important; border-color: rgba(151, 184, 231, .23) !important; border-radius: 10px !important;
            }
            [data-testid="stForm"] [data-baseweb="select"] svg { fill: #94c8ff; }
            [data-testid="stForm"] button[kind="primary"] {
                border: 1px solid rgba(151, 210, 255, .3); border-radius: 11px; padding: .72rem 1rem; font-size: 1rem;
                font-weight: 750; background: linear-gradient(100deg, #287bf1, #54b6ff); box-shadow: 0 10px 20px rgba(32, 116, 240, .22);
            }
            [data-testid="stForm"] button[kind="primary"]:hover { background: linear-gradient(100deg, #398bf8, #66c3ff); }
            .result-card {
                margin-top: 1.3rem; padding: 1.7rem 1.85rem; border: 1px solid rgba(148, 208, 255, .3); border-radius: 20px;
                background: linear-gradient(115deg, #0b5fbe, #173d85); box-shadow: 0 20px 38px rgba(0, 0, 0, .25);
            }
            .result-card .caption { margin: 0; color: #d5eaff; font-size: .76rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
            .result-card h2 { margin: .25rem 0 .45rem; font-size: 2.45rem; }
            .result-card .description { margin: 0; color: #e4f1ff; }
            [data-testid="stMetric"] {
                border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.1rem;
                background: linear-gradient(145deg, rgba(17, 53, 105, .82), rgba(7, 27, 60, .82));
            }
            [data-testid="stMetricLabel"] p { color: #a9c0e2 !important; font-weight: 650; }
            [data-testid="stMetricValue"] { color: #f1f7ff !important; }
            .note { color: #9fb4d3; font-size: .9rem; line-height: 1.6; margin-top: 1.25rem; }
            .note code { color: #9bd3ff; background: rgba(87, 165, 255, .12); padding: .1rem .35rem; border-radius: 5px; }
            @media (max-width: 700px) {
                .block-container { padding: 1.1rem .85rem 2.2rem; }
                .hero { padding: 2rem 1.45rem; } .hero::after { opacity: .45; }
                [data-testid="stForm"] { padding: 1.1rem; } .prediction-intro { align-items: flex-start; flex-direction: column; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <style>
            :root {
                --ink: #1e1d3d; --muted: #6d6b86; --lavender: #f6f4ff;
                --line: #e6e2f4; --purple: #7453e8; --blue: #3e8df5;
                --card: #ffffff; --soft-purple: #f0edff;
            }
            .stApp { background: linear-gradient(135deg, #fbfaff 0%, #f4f1ff 55%, #f8f7ff 100%) !important; color: var(--ink) !important; }
            [data-testid="stHeader"] { background: rgba(255, 255, 255, .92) !important; border-bottom: 1px solid var(--line); }
            [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid var(--line) !important; }
            [data-testid="stSidebar"] * { color: var(--ink) !important; }
            [data-testid="stSidebar"] > div:first-child { padding-top: 1.15rem; }
            [data-testid="stSidebar"] .stRadio label { margin: .12rem 0; padding: .48rem .55rem; border-radius: 10px; }
            [data-testid="stSidebar"] .stRadio label:hover { background: #f3f0ff; }
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
            .block-container { max-width: 1280px; padding-top: 1.35rem; padding-bottom: 3.5rem; }
            h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -.03em; }
            p, .stMarkdown { color: var(--muted); }
            .app-header {
                display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 2.1rem;
                padding: 1rem 1.2rem; border: 1px solid rgba(229, 225, 244, .9); border-radius: 18px; background: rgba(255, 255, 255, .88);
                box-shadow: 0 10px 26px rgba(67, 48, 128, .06);
            }
            .brand-lockup { display: flex; align-items: center; gap: .8rem; }
            .app-mark {
                display: grid; place-items: center; width: 2.6rem; height: 2.6rem; border-radius: 12px; color: #fff;
                font-size: .78rem; font-weight: 850; letter-spacing: -.03em; background: linear-gradient(135deg, #348dfa, #8050e8);
                box-shadow: 0 8px 16px rgba(95, 80, 218, .22);
            }
            .brand-name { color: #242145; font-size: 1.08rem; font-weight: 800; letter-spacing: -.03em; }
            .brand-caption { margin-top: .12rem; color: #8a88a1; font-size: .77rem; }
            .header-status { color: #625e80; font-size: .78rem; font-weight: 700; }
            .header-status::before { content: ""; display: inline-block; width: .46rem; height: .46rem; margin-right: .42rem; border-radius: 50%; background: #5dcb9a; box-shadow: 0 0 0 .2rem rgba(93, 203, 154, .14); }
            .page-intro { margin: 0 0 1.4rem; }
            .page-kicker { margin-bottom: .38rem; color: #7656e4; font-size: .75rem; font-weight: 850; letter-spacing: .12em; text-transform: uppercase; }
            .page-intro h1 { margin: 0 0 .45rem; font-size: clamp(2rem, 3.4vw, 3rem); line-height: 1.1; }
            .page-intro p { max-width: 40rem; margin: 0; font-size: 1rem; line-height: 1.65; }
            .dashboard-card, .surface-card {
                box-sizing: border-box; height: 100%; padding: 1.35rem; border: 1px solid var(--line); border-radius: 18px;
                background: var(--card); box-shadow: 0 12px 30px rgba(54, 39, 111, .07);
            }
            .feature-icon { display: grid; place-items: center; width: 2.25rem; height: 2.25rem; margin-bottom: .85rem; border-radius: 9px; color: #684bd4; background: #f0edff; font-size: .78rem; font-weight: 850; }
            .dashboard-card h3, .surface-card h3 { margin: 0 0 .45rem; font-size: 1.02rem; }
            .dashboard-card p, .surface-card p { margin: 0; color: var(--muted); line-height: 1.58; }
            .quick-stat { padding: 1rem 1.05rem; border: 1px solid #e6e2f4; border-radius: 15px; background: rgba(255, 255, 255, .76); }
            .quick-stat strong { display: block; color: #2a274a; font-size: 1.22rem; letter-spacing: -.04em; }
            .quick-stat span { color: #85829d; font-size: .78rem; }
            .form-layout-title { display: none; }
            [data-testid="stForm"] {
                padding: 0 1.35rem 1.35rem !important; border: 1px solid var(--line) !important; border-radius: 18px !important;
                background: #ffffff !important; box-shadow: 0 12px 30px rgba(54, 39, 111, .07) !important;
            }
            .form-gradient-header { margin: 0 -1.35rem 1.3rem; padding: 1.1rem 1.35rem; border-radius: 17px 17px 0 0; background: linear-gradient(105deg, #368cf6, #7551e8); }
            .form-gradient-header h2 { margin: 0 0 .18rem; color: #fff !important; font-size: 1.18rem; }
            .form-gradient-header p { margin: 0; color: rgba(255, 255, 255, .86); font-size: .84rem; }
            [data-testid="stForm"] label, [data-testid="stForm"] [data-testid="stWidgetLabel"] p { color: #343151 !important; font-size: .84rem; font-weight: 700; }
            [data-testid="stForm"] [data-baseweb="select"] > div,
            [data-testid="stForm"] input {
                min-height: 2.45rem; border-color: #ddd8ee !important; border-radius: 9px !important; background: #ffffff !important; color: #2e2b49 !important;
                box-shadow: none !important;
            }
            [data-testid="stForm"] [data-baseweb="select"] > div:hover,
            [data-testid="stForm"] input:focus { border-color: #9176ed !important; }
            [data-testid="stForm"] [data-baseweb="select"] svg { fill: #7656e4; }
            .additional-inputs { margin: .45rem 0 .25rem; padding-top: 1.05rem; border-top: 1px solid #eeeaf6; color: #716d8b; font-size: .75rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
            [data-testid="stForm"] button[kind="primary"] {
                width: 100%; min-height: 2.75rem; margin-top: .75rem; border: 0 !important; border-radius: 10px !important;
                background: linear-gradient(100deg, #6946dd, #a14ee7) !important; box-shadow: 0 10px 18px rgba(119, 75, 218, .24) !important;
                color: #fff !important; font-size: .95rem; font-weight: 800;
            }
            [data-testid="stForm"] button[kind="primary"]:hover { background: linear-gradient(100deg, #5e3bd0, #9442d8) !important; transform: translateY(-1px); }
            .comparison-card { overflow: hidden; padding: 0; }
            .comparison-heading { padding: 1.2rem 1.3rem 1rem; border-bottom: 1px solid #eeeaf6; }
            .comparison-heading h2 { margin: 0 0 .28rem; font-size: 1.16rem; }
            .comparison-heading p { margin: 0; color: #85819d; font-size: .82rem; }
            .comparison-table { width: 100%; border-collapse: collapse; font-size: .84rem; }
            .comparison-table th { padding: .75rem 1.3rem; color: #85819d; font-size: .69rem; font-weight: 850; letter-spacing: .07em; text-align: left; text-transform: uppercase; background: #faf9ff; }
            .comparison-table td { padding: .82rem 1.3rem; color: #4a4765; border-top: 1px solid #f0edf7; font-weight: 650; }
            .comparison-table td strong { color: #6946dd; }
            .comparison-table .not-available { color: #aaa6bb; font-weight: 600; }
            .performance-chart { margin: 1.15rem 1.3rem 1.3rem; padding: 1rem; border: 1px solid #ece8f5; border-radius: 13px; background: linear-gradient(145deg, #fcfbff, #f6f3ff); }
            .chart-label { display: flex; justify-content: space-between; gap: .75rem; margin-bottom: .68rem; color: #4d4968; font-size: .8rem; font-weight: 800; }
            .chart-label span:last-child { color: #7656e4; }
            .bar-track { height: .54rem; overflow: hidden; border-radius: 999px; background: #e7e2f4; }
            .bar-fill { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #3b8ef7, #8a55e7); }
            .chart-legend { display: flex; align-items: center; gap: .45rem; margin-top: .8rem; color: #89859e; font-size: .72rem; }
            .chart-dot { width: .48rem; height: .48rem; border-radius: 50%; background: linear-gradient(135deg, #3b8ef7, #8a55e7); }
            .model-note { margin: 0 1.3rem 1.25rem; color: #9793a8; font-size: .72rem; line-height: 1.45; }
            .result-card { margin-top: 1.25rem; padding: 1.5rem 1.65rem; border: 1px solid #dcd5fb; border-radius: 18px; background: linear-gradient(115deg, #f5f1ff, #edf5ff); box-shadow: 0 12px 26px rgba(72, 48, 144, .08); }
            .result-card .caption { margin: 0; color: #7352d9; font-size: .73rem; font-weight: 850; letter-spacing: .1em; text-transform: uppercase; }
            .result-card h2 { margin: .25rem 0 .4rem; color: #2d2856 !important; font-size: 2.25rem; }
            .result-card .description { margin: 0; color: #66637f; }
            [data-testid="stMetric"] { padding: 1rem 1.05rem; border: 1px solid var(--line); border-radius: 15px; background: #ffffff; box-shadow: 0 9px 20px rgba(54, 39, 111, .05); }
            [data-testid="stMetricLabel"] p { color: #79758f !important; font-weight: 700; }
            [data-testid="stMetricValue"] { color: #2d2856 !important; }
            .note { color: #85819d; font-size: .84rem; line-height: 1.6; }
            .note code { color: #6549cb; background: #f0edff; padding: .08rem .3rem; border-radius: 4px; }
            @media (max-width: 700px) {
                .block-container { padding: 1rem .8rem 2.2rem; }
                .app-header { margin-bottom: 1.45rem; padding: .85rem .9rem; } .header-status { display: none; }
                [data-testid="stForm"] { padding: 0 1rem 1rem !important; } .form-gradient-header { margin: 0 -1rem 1.1rem; padding: 1rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_app_header() -> None:
    st.markdown(
        """
        <header class="app-header">
          <div class="brand-lockup">
            <div class="app-mark">AI</div>
            <div><div class="brand-name">AI Salary Predictor</div><div class="brand-caption">Employee compensation intelligence</div></div>
          </div>
          <div class="header-status">Model ready</div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def show_home() -> None:
    st.markdown(
        """
        <section class="page-intro">
          <div class="page-kicker">Salary intelligence</div>
          <h1>Make compensation decisions with confidence.</h1>
          <p>Use the trained model to turn an employee profile into a clear salary estimate in just a few steps.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    metrics = st.columns(3)
    summary = (("9", "employee inputs"), ("48", "model-ready features"), ("96.35%", "explained variance"))
    for column, (value, label) in zip(metrics, summary):
        column.markdown(f"<div class='quick-stat'><strong>{value}</strong><span>{label}</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1.35rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<h2 style='font-size:1.35rem; margin:0 0 .25rem'>Everything you need, in one focused workspace</h2>",
        unsafe_allow_html=True,
    )
    first, second, third = st.columns(3)
    cards = [
        ("01", "Guided inputs", "Choose from options learned by the supplied training artifact for a clean, consistent profile."),
        ("02", "Trusted pipeline", "The saved encoder and scaler are used without refitting, so the model input remains consistent."),
        ("03", "Clear estimate", "Receive a precise, formatted salary estimate from the saved Linear Regression model."),
    ]
    for column, (index, heading, text) in zip((first, second, third), cards):
        with column:
            st.markdown(
                f'<div class="dashboard-card"><div class="feature-icon">{index}</div><h3>{heading}</h3><p>{text}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown("<p class='note'>Open <strong>Salary Prediction</strong> in the sidebar to enter employee information and generate an estimate.</p>", unsafe_allow_html=True)


def show_model_comparison() -> None:
    """Render the supplied model's verified evaluation metrics without fabricating a baseline."""
    st.markdown(
        """
        <section class="dashboard-card comparison-card">
          <div class="comparison-heading">
            <h2>Model Performance Comparison</h2>
            <p>Verified evaluation metrics for the saved model.</p>
          </div>
          <table class="comparison-table">
            <thead><tr><th>Metric</th><th>Enhanced</th><th>Baseline</th></tr></thead>
            <tbody>
              <tr><td>R&sup2; Score</td><td><strong>0.9635</strong></td><td class="not-available">Not provided</td></tr>
              <tr><td>MAE</td><td><strong>5,436.10</strong></td><td class="not-available">Not provided</td></tr>
            </tbody>
          </table>
          <div class="performance-chart">
            <div class="chart-label"><span>Enhanced R&sup2; score</span><span>96.35%</span></div>
            <div class="bar-track"><span class="bar-fill" style="width:96.35%"></span></div>
            <div class="chart-legend"><span class="chart-dot"></span><span>Enhanced model explained variance</span></div>
          </div>
          <p class="model-note">A separate baseline model or baseline evaluation was not included with this project, so its metrics remain intentionally unavailable.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_prediction_form(artifacts: ModelArtifacts) -> None:
    st.markdown(
        """
        <section class="page-intro">
          <div class="page-kicker">Salary prediction</div>
          <h1>Build an employee profile.</h1>
          <p>Enter the employee details to generate a salary estimate with the saved prediction model.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    form_column, comparison_column = st.columns((1.16, 0.84), gap="large")
    with form_column:
        with st.form("salary_prediction_form", clear_on_submit=False):
            st.markdown(
                """
                <div class="form-gradient-header">
                  <h2>Employee Details</h2>
                  <p>Complete the profile below to prepare an estimate.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            left, right = st.columns(2)
            with left:
                experience_years = st.number_input(
                    "Years of Experience", min_value=0.0, value=1.0, step=0.5, help="Enter a non-negative number of years."
                )
                job_title = st.selectbox("Job Title", category_options(artifacts, "job_title"), format_func=str)
                industry = st.selectbox("Industry", category_options(artifacts, "industry"), format_func=str)
            with right:
                education_level = st.selectbox(
                    "Education Level", category_options(artifacts, "education_level"), format_func=str
                )
                location = st.selectbox("Location Tier", category_options(artifacts, "location"), format_func=str)
                company_size = st.selectbox(
                    "Company Size", category_options(artifacts, "company_size"), format_func=str
                )

            st.markdown("<div class='additional-inputs'>Additional model inputs</div>", unsafe_allow_html=True)
            supporting_left, supporting_middle, supporting_right = st.columns(3)
            with supporting_left:
                skills_count = st.number_input(
                    "Skills Count", min_value=0, value=1, step=1, help="Enter a non-negative whole number."
                )
            with supporting_middle:
                remote_work = st.selectbox("Remote Work", category_options(artifacts, "remote_work"), format_func=str)
            with supporting_right:
                certifications = st.number_input(
                    "Certifications", min_value=0, value=0, step=1, help="Enter a non-negative whole number."
                )
            submitted = st.form_submit_button("Predict Salary", type="primary", width="stretch")

    with comparison_column:
        show_model_comparison()

    if not submitted:
        return

    values = {
        "job_title": job_title,
        "experience_years": experience_years,
        "education_level": education_level,
        "skills_count": skills_count,
        "industry": industry,
        "company_size": company_size,
        "location": location,
        "remote_work": remote_work,
        "certifications": certifications,
    }
    numeric_values = (experience_years, skills_count, certifications)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in numeric_values):
        st.error("Experience, skills count, and certifications must be finite non-negative values.")
        return

    try:
        salary = predict_salary(values, artifacts)
    except Exception as error:  # Display a helpful message without exposing a stack trace in the UI.
        st.error(f"The salary prediction could not be completed: {error}")
        return

    st.markdown(
        f"""
        <section class="result-card">
          <p class="caption">Estimated salary</p>
          <h2>{salary:,.2f}</h2>
          <p class="description">Based on the employee details provided, the trained model estimates this salary.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_about(artifacts: ModelArtifacts) -> None:
    st.markdown(
        """
        <section class="page-intro">
          <div class="page-kicker">About</div>
          <h1>Model transparency, at a glance.</h1>
          <p>Review the evaluation metrics and the saved pipeline behind each salary estimate.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    metrics_row = st.columns(4)
    for column, (label, value) in zip(metrics_row, MODEL_METRICS.items()):
        column.metric(label, value)

    info_left, info_right = st.columns((1.15, 1))
    with info_left:
        st.markdown(
            """
            <div class="dashboard-card">
              <h3>Training summary</h3>
              <p><strong>Algorithm:</strong> Linear Regression<br>
              <strong>Original dataset:</strong> 250,000 records<br>
              <strong>Input features:</strong> 9 employee attributes<br>
              <strong>Model input:</strong> 48 scaled features after one-hot encoding</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_right:
        st.markdown(
            """
            <div class="dashboard-card">
              <h3>How to read the score</h3>
              <p>An R² score of 0.9635 means the model explains approximately <strong>96.35%</strong> of the variance in salary. It is not a classification accuracy score.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<p class='note'>Loaded artifact: <code>{artifact_path().name}</code> · "
        f"fitted categories: {sum(len(group) for group in artifacts.encoder.categories_)} · "
        f"feature schema: {artifacts.model.n_features_in_} columns.</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    try:
        artifacts = load_artifacts(str(artifact_path()))
    except Exception as error:
        st.error("The trained model artifacts could not be loaded.")
        st.exception(error)
        st.stop()

    with st.sidebar:
        st.markdown("## AI Salary Predictor")
        st.caption("Employee prediction workspace")
        page = st.radio("Navigate", ("Home", "Salary Prediction", "About"))
        st.divider()
        st.caption("Uses the saved model, encoder, and scaler from this project.")

    show_app_header()

    if page == "Home":
        show_home()
    elif page == "Salary Prediction":
        show_prediction_form(artifacts)
    else:
        show_about(artifacts)


if __name__ == "__main__":
    main()
