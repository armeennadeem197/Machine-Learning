# Employee Salary Prediction

A Streamlit dashboard for the supplied Employee Salary Prediction model. It loads the existing fitted `LinearRegression`, `OneHotEncoder`, and `StandardScaler` from `employee_salary_model.joblib`; it does **not** retrain or refit any object.

## Run locally

Use a Python 3.10–3.13 environment (the saved model was created with
scikit-learn 1.6.1), then install the project dependencies and start Streamlit.
For example, with Python 3.13 installed:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Use another installed 3.10–3.13 version in place of `3.13` if needed. If `py`
is not available, create the environment with the matching Python executable.

## Model artifacts

The app looks for the supplied artifact in either of these locations:

```text
employee_salary_model.joblib
artifacts/employee_salary_model.joblib
```

The current artifact already contains these fitted training objects and schema keys:

- `model` — `LinearRegression`
- `encoder` — fitted `OneHotEncoder` with `handle_unknown="ignore"`
- `scaler` — fitted `StandardScaler`
- `categorical_col` and `numeric_col` — the original preprocessing order

The app reads dropdown values from `encoder.categories_`, concatenates the three numeric fields before the 45 encoded categorical fields, and then applies the saved scaler. It validates that the resulting input has the exact 48 features expected by both the saved scaler and model. Salary is shown as the model's numeric target value with thousands separators; the project materials do not identify a currency or payment period, so the app does not assume either.

## Dataset and evaluation

- Original dataset size: 250,000 records
- Raw input features: 9
- Algorithm: Linear Regression
- MAE: 5,436.10
- MSE: 50,773,076.35
- RMSE: 7,125.52
- R² score: 0.9635 — approximately 96.35% explained variance, not model accuracy

## Notes

No dataset or notebook was present in the supplied workspace, and the existing artifact already contains the fitted preprocessing objects. Therefore no artifact export code or dataset modification is required for this app. Keep the Joblib artifact next to `app.py` (as it is now) or move it intact to `artifacts/`.

## Appearance

The committed `.streamlit/config.toml` sets the dark-navy, blue-accented theme
and professional typography. The app adds the light result and information
cards while retaining Streamlit's responsive layout.
