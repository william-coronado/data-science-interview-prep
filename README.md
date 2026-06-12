# Data Science Interview Prep Notebook

An end-to-end case study for rehearsing data science technical interviews and
take-home coding tests. It works a single, realistic problem — fraud
detection for a fictional fintech, "Northwind Pay" — through six topic areas
that commonly come up in DS interview loops, with an emphasis on
**production-grade, tested code** alongside the analysis.

## Contents

| File | Purpose |
|---|---|
| `northwind_fraud_casestudy.ipynb` | The case-study notebook (run top to bottom) |
| `fraud_utils.py` | Reusable module: synthetic data generators, validation, preprocessing, evaluation metrics, fairness metrics, and a class-based `FraudModelPipeline` |
| `test_fraud_utils.py` | `pytest` suite covering happy paths, edge cases, and error contracts for `fraud_utils.py` |
| `build_notebook.py` | Regenerates `northwind_fraud_casestudy.ipynb` from the narrative + the modules above |

The notebook **writes `fraud_utils.py` and `test_fraud_utils.py` to disk and
runs `pytest` in a subprocess**, so the same code shown and explained in the
narrative is the code that is actually under test — mirroring how a real
project separates notebook narrative from reusable, tested modules.

## Topics covered

1. **Statistical Modelling & Hypothesis Testing** — A/B test design, sample
   size / power analysis, two-proportion z-test, chi-square, t-test vs
   Mann-Whitney, confidence intervals, a frequentist vs Bayesian comparison,
   and the Poisson/binomial/normal distributions behind the data.
2. **Supervised Learning** — logistic regression, random forest, XGBoost and
   LightGBM in scikit-learn pipelines; ROC-AUC/PR-AUC/precision/recall/F1 with
   commentary on when each matters; stratified k-fold vs time-series CV;
   handling class imbalance (class weights, SMOTE, threshold tuning);
   permutation importance and SHAP; regularisation (L1/L2/ElasticNet) and
   hyperparameter tuning with Optuna.
3. **Production-Ready Python** — a class-based, type-hinted, docstring'd
   pipeline (`FraudModelPipeline`) with `ColumnTransformer`-based
   preprocessing, structured logging, explicit error handling for edge cases
   (empty frames, missing columns, bad types), and a `pytest` suite runnable
   from the command line.
4. **Data Wrangling & Preprocessing** — `groupby`/`merge`/`pivot`/`melt`/`apply`
   on realistic business questions; imputation strategies and their
   trade-offs against non-random missingness; ordinal vs one-hot vs target
   encoding; datetime feature engineering; and a **deliberate data-leakage
   trap** that is diagnosed and fixed.
5. **Model Interpretability & Ethics** — SHAP explanations (including a
   plain-English version for non-technical stakeholders), fairness metrics
   (demographic parity, equalised odds, disparate impact ratio), bias
   detection and a mitigation example, and a model card template.
6. **Data Visualisation for Communication** — matplotlib/seaborn charts
   chosen by question type (distribution, comparison, relationship,
   composition), each titled with the "so what," plus one interactive plotly
   chart.

## Setup

Requires Python 3.10+. From this directory:

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Your shell prompt should now show `(.venv)`. From here on, `python` and
`pip` refer to the versions inside the virtual environment. To leave it
later, run `deactivate`.

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**macOS only:** XGBoost and LightGBM need the OpenMP runtime, which is not
installed by pip:

```bash
brew install libomp
```

> The commands below assume an activated environment (`python`/`pip`). If
> you prefer not to activate, prefix each command with `.venv/bin/`, e.g.
> `.venv/bin/jupyter lab ...`.

## Running

With the virtual environment activated:

Open and run the notebook top to bottom:

```bash
jupyter lab northwind_fraud_casestudy.ipynb
```

Or execute it headlessly end-to-end (~8 minutes):

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 northwind_fraud_casestudy.ipynb
```

Run the unit tests independently:

```bash
python -m pytest -q
```

To regenerate the notebook from source after editing `build_notebook.py`,
`fraud_utils.py`, or `test_fraud_utils.py`:

```bash
python build_notebook.py
```

## Adapting this for your own prep

The synthetic data, pipeline, and metrics all live in `fraud_utils.py` and
the narrative cells in `build_notebook.py`. To rehearse a different domain
(e.g. churn, pricing, demand forecasting), swap the data generators and
feature lists — the statistical, ML, production, wrangling, ethics, and
visualisation sections are structured generically enough to carry over.
