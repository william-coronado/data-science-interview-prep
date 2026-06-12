# Concepts Guide: A Beginner's Walkthrough

This guide explains, in plain language, the core data science concepts used in
this repository's case study (`northwind_fraud_casestudy.ipynb`,
`fraud_utils.py`, `build_notebook.py`). It's meant to be read **alongside**
the notebook — each section below points to where the idea shows up in the
code, and builds up the intuition first so the code reads as an
implementation of an idea you already understand, not a black box.

If you're new to data science, read the sections in order. If you already
know some of this, jump straight to the topics flagged in the table of
contents — they get extra depth because they're the ones most often probed
in interviews.

## Table of contents

1. [A/B testing & experiment design](#1-ab-testing--experiment-design) ⭐
2. [Classification models in scikit-learn pipelines](#2-classification-models-in-scikit-learn-pipelines) ⭐
   - [Logistic Regression](#21-logistic-regression)
   - [Random Forest](#22-random-forest)
   - [XGBoost](#23-xgboost)
   - [LightGBM](#24-lightgbm)
   - [Why "pipelines"? `ColumnTransformer` + leakage](#25-why-pipelines-columntransformer--leakage)
3. [Evaluation metrics: ROC-AUC, PR-AUC, precision, recall, F1](#3-evaluation-metrics-roc-auc-pr-auc-precision-recall-f1) ⭐
4. [Cross-validation](#4-cross-validation) ⭐
5. [Handling class imbalance (incl. SMOTE)](#5-handling-class-imbalance-incl-smote) ⭐
6. [Model interpretability: permutation importance & SHAP](#6-model-interpretability-permutation-importance--shap) ⭐
7. [Regularisation & hyperparameter tuning](#7-regularisation--hyperparameter-tuning)
8. [Fairness metrics](#8-fairness-metrics)
9. [Quick-reference glossary](#9-quick-reference-glossary)
10. [Suggested reading order through the notebook](#10-suggested-reading-order-through-the-notebook)

---

## 1. A/B testing & experiment design

**The business question.** Northwind Pay wants to know whether adding a
"step-up authentication" step at checkout reduces fraud — and at what cost to
checkout speed (`build_notebook.py`, Section 1).

### What is an A/B test?

An A/B test (a.k.a. a randomised controlled experiment) splits users into two
random groups:

- **Control (A)** — sees the existing experience.
- **Treatment (B)** — sees the new experience (here, the extra
  authentication step).

Because the split is *random*, any difference in outcomes between the groups
can be attributed to the change itself — not to who happened to land in each
group. This is what lets us say "X **caused** Y" rather than just "X is
correlated with Y."

`fraud_utils.generate_ab_experiment()` simulates exactly this: a `control`
and `treatment` group, each with a `converted` flag, an `is_fraud` flag, and
a `checkout_seconds` value.

### Power analysis: how big does the experiment need to be?

Before running an experiment, you should ask: *if there really is an effect
of the size we care about, how likely are we to detect it?* That likelihood
is **statistical power**, and the calculation that tells you the required
sample size is **power analysis**.

Four numbers are linked together — fix any three and the fourth is
determined:

| Term | Meaning |
|---|---|
| **Baseline rate** (`p_control`) | The current value of the metric (e.g. 2.1% fraud) |
| **MDE** (minimum detectable effect) | The smallest change you care about being able to detect (e.g. a 0.8 pp drop) |
| **Significance level (α)** | The false-positive rate you're willing to accept (usually 0.05) |
| **Power (1 − β)** | The probability of detecting the effect if it's real (usually 0.80) |

`fraud_utils.required_sample_size_two_proportions()` (`fraud_utils.py:458`)
implements the standard formula for two proportions and returns "how many
customers do I need **per group**?" Run this *before* looking at any data —
running it afterwards to justify a result you already like is a form of
p-hacking.

> **Why this matters:** an under-powered test that finds "no significant
> difference" is uninformative — you can't tell whether there's truly no
> effect, or whether you just didn't have enough data to see it.

### Picking the right significance test

The right test depends on the **type of outcome variable**:

| Outcome type | Example in this repo | Test(s) used |
|---|---|---|
| Binary (yes/no) | `is_fraud` (did this transaction turn out to be fraud?) | **Two-proportion z-test**, **chi-square test of independence** |
| Continuous, skewed | `checkout_seconds` | **Welch's t-test** (means) and **Mann-Whitney U** (distributions, no normality assumption) |

- **Two-proportion z-test** (`fraud_utils.two_proportion_ztest()`,
  `fraud_utils.py:424`) — compares two proportions (e.g. fraud rate in
  control vs. treatment) using a z-statistic. The **chi-square test** on the
  2×2 contingency table (`group` × `is_fraud`) answers the same question and
  will give a very similar p-value — they're mathematically related for 2×2
  tables.
- **Welch's t-test** (`scipy.stats.ttest_ind(..., equal_var=False)`) compares
  *means* and assumes the data are roughly normal. **Checkout time is
  right-skewed** (most checkouts are fast, a few are very slow), so the mean
  can be pulled around by outliers.
- **Mann-Whitney U test** compares whole *distributions* without assuming
  normality — it's the safer headline number on skewed data. Both tests are
  reported; when they disagree, trust the non-parametric one.

### Confidence intervals

A **confidence interval (CI)** gives a *range* of plausible values for the
true effect, not just a single point estimate. The notebook computes a 95%
Wald CI for the difference in fraud rates
(`(p_treatment - p_control) ± 1.96 × SE`). If the entire interval is below
zero, you can be confident the *direction* of the effect (a reduction) is
real, and the width of the interval tells you the plausible *magnitude*.

> **What a p-value actually means** (a common interview trap): a p-value is
> **not** "the probability the treatment works." It is *the probability of
> seeing data this extreme (or more extreme) if the treatment had **no**
> effect*. A small p-value means "the no-effect story fits the data badly" —
> it says nothing directly about effect size. That's what the CI is for.

### Frequentist vs. Bayesian framing

The frequentist tests above produce a p-value and a CI. A **Bayesian**
approach instead asks directly: *given the data, what's the probability the
treatment is better?* The notebook does this with a simple **Beta-Binomial**
model — starting from an uninformative `Beta(1,1)` prior over each group's
fraud rate, updating it with the observed counts, and then simulating from
the posterior distributions to compute:

- `P(treatment fraud rate < control fraud rate)` — directly interpretable as
  "there's an 98% probability the treatment reduces fraud."
- The **expected reduction** — the average difference between posterior
  draws.

This is often more useful in a stakeholder conversation, because it answers
the question people actually ask ("how likely is it to work, and by how
much?") rather than the indirect frequentist framing.

### The distributions behind it all

Three distributions show up repeatedly and are worth being able to recognise:

- **Poisson** — counts of events in a fixed window (e.g. transactions per
  customer per 24h). Defined by one parameter, λ (the mean rate).
- **Binomial** — number of "successes" out of a fixed number of independent
  trials with the same probability (e.g. how many of 50 transactions are
  foreign).
- **Normal (Gaussian)** — the familiar bell curve; many naturally-occurring
  continuous quantities are approximately normal *after* a transformation
  (here, `log(1 + amount)` is roughly normal even though `amount` itself is
  skewed/log-normal).

---

## 2. Classification models in scikit-learn pipelines

The case study trains **four classifiers** to predict `is_fraud` (a binary
label), all wrapped in the same `FraudModelPipeline` class
(`fraud_utils.py:602`) so they're trained and evaluated identically
(`build_notebook.py`, Section 2.2). Understanding *why* each model is
included — and what makes it different from the others — is one of the most
common interview themes.

### 2.1 Logistic Regression

**The idea.** Logistic regression is a *linear* model for binary outcomes. It
computes a weighted sum of the input features (the "linear predictor" or
"log-odds") and then squashes that sum into a probability between 0 and 1
using the **sigmoid function**:

```
log-odds = b0 + b1*x1 + b2*x2 + ... + bn*xn
P(y=1) = 1 / (1 + exp(-log-odds))      <- the sigmoid
```

- Each coefficient `b_i` tells you how the log-odds of the outcome change for
  a one-unit increase in `x_i`, holding everything else fixed. This is what
  makes logistic regression **interpretable** — you can read the
  coefficients directly (after scaling features so they're comparable).
- It is trained by maximising the likelihood of the observed labels
  (equivalently, minimising "log loss" / cross-entropy).
- It's a **linear** model in the sense that the *log-odds* are a linear
  function of the inputs — it can still produce curved decision boundaries
  if you feed it interaction or polynomial features, but on its own it
  assumes additive, linear relationships in log-odds space.

**Why it needs scaling.** Because the model fits one coefficient per feature
on a shared additive scale, features with very different magnitudes (e.g.
`amount` in dollars vs. `is_foreign` as 0/1) will have wildly different
coefficient sizes purely due to units — and regularisation (see [Section
7](#7-regularisation--hyperparameter-tuning)) penalises large coefficients
indiscriminately. **Standard-scaling** (mean 0, std 1) puts every feature on
a comparable footing. In `model_specs` (`build_notebook.py:386`), logistic
regression is the only model trained with `scale=True`.

**`class_weight="balanced"`.** Fraud is rare (~1.5%). Without intervention,
the model can get a very low loss just by predicting "not fraud" for
everything. `class_weight="balanced"` re-weights the loss so that
mis-classifying a minority-class (fraud) example counts more — see [Section
5](#5-handling-class-imbalance-incl-smote).

**Strengths / weaknesses:**
- ✅ Fast, interpretable (coefficients ≈ feature effect on log-odds), a great
  baseline, calibrated probabilities tend to be reasonable out of the box.
- ❌ Assumes additive linear relationships in log-odds space; can't capture
  complex feature interactions or non-linear effects without manual feature
  engineering.

### 2.2 Random Forest

**The idea.** A **decision tree** repeatedly splits the data on the feature
and threshold that best separates the classes (e.g. "is `amount` > $200?"),
building a tree of yes/no questions until each "leaf" is fairly pure. A
single tree, grown deep, tends to **overfit** — it memorises the training
data's quirks.

A **Random Forest** is an *ensemble* of many such trees, where each tree is
trained on:

- A **bootstrap sample** of the rows (sampling with replacement) — this is
  "bagging" (bootstrap aggregating).
- A **random subset of features** considered at each split.

The forest's prediction is the **average** (for probabilities) or **majority
vote** (for hard labels) across all trees. Averaging many trees that each
overfit *differently* cancels out a lot of the overfitting — the random
feature subsets also decorrelate the trees so they don't all make the same
mistakes.

In `model_specs` (`build_notebook.py:388`):
```python
RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1,
                        class_weight="balanced", random_state=SEED)
```
- `n_estimators=300` — number of trees in the forest.
- `max_depth=12` — caps how deep each tree can grow (controls
  overfitting/complexity).
- `n_jobs=-1` — train trees in parallel using all CPU cores (trees are
  independent of each other, unlike boosting — see below).

**Why no scaling.** Trees split on `feature > threshold`. The *threshold*
adapts to whatever scale the feature is on, so multiplying a feature by 1000
doesn't change which splits are chosen. Tree-based models are scale-invariant
— `scale=False` for Random Forest, XGBoost and LightGBM in this repo.

**Strengths / weaknesses:**
- ✅ Captures non-linear relationships and interactions automatically;
  robust to outliers and irrelevant features; minimal preprocessing.
- ❌ Less interpretable than logistic regression (though you can extract
  feature importances); can be memory/compute heavy with many deep trees;
  probability estimates are often less well-calibrated.

### 2.3 XGBoost

**The idea: gradient boosting.** Where Random Forest builds many *independent*
trees and averages them, **gradient boosting** builds trees **sequentially**,
where each new tree is trained to correct the *errors (residuals)* of the
trees built so far. Concretely:

1. Start with a simple prediction (e.g. the average).
2. Compute the gradient of the loss function with respect to each
   prediction — roughly, "how wrong, and in which direction, is the current
   model for this example?"
3. Fit a new (small) tree to predict that gradient.
4. Add a scaled-down (`learning_rate`) version of that tree's predictions to
   the running total.
5. Repeat for `n_estimators` rounds.

Each tree is shallow (`max_depth`) and contributes only a small correction
(`learning_rate`), but hundreds of them stacked together can fit very complex
functions — while being regularised by the fact that each step is small.

**XGBoost** ("eXtreme Gradient Boosting") is a fast, regularised
implementation of this idea with extra engineering for handling missing
values natively, parallel/histogram-based split-finding, and built-in L1/L2
regularisation on leaf weights.

In `model_specs` (`build_notebook.py:391`):
```python
XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
              subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
              scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=SEED)
```
- `learning_rate=0.05` — small step size; needs more `n_estimators` to
  compensate, but generalises better than a large step size.
- `max_depth=5` — each tree is shallow (a "weak learner").
- `subsample=0.9` / `colsample_bytree=0.9` — each tree sees 90% of rows /
  columns (sampled), adding randomness that reduces overfitting (similar
  spirit to bagging).
- `eval_metric="aucpr"` — area under the precision-recall curve, chosen
  because of class imbalance (see [Section 3](#3-evaluation-metrics-roc-auc-pr-auc-precision-recall-f1)).
- `scale_pos_weight` — ratio of negative to positive examples
  (`(y==0).sum() / (y==1).sum())`), XGBoost's equivalent of `class_weight`:
  it up-weights the gradient contribution of the minority (fraud) class.

### 2.4 LightGBM

**The idea.** LightGBM is *also* a gradient-boosting implementation (same
sequential-correction idea as XGBoost), but with different engineering
choices that usually make it faster on large/wide tabular data:

- **Histogram-based splitting** — features are bucketed into discrete bins
  before training, so finding the best split is a fast lookup rather than
  scanning every unique value (XGBoost can also do this, but it was
  LightGBM's signature innovation).
- **Leaf-wise (best-first) tree growth** — instead of growing a tree
  level-by-level, LightGBM always splits whichever leaf will reduce the loss
  the most, producing deeper, more asymmetric trees for the same node budget.
  This tends to reach lower loss faster, but can overfit small datasets more
  easily — hence `num_leaves` (max leaves per tree) is the primary
  complexity control, rather than `max_depth`.
- Native support for categorical features (though this repo one-hot encodes
  via the shared preprocessor for consistency across models).

In `model_specs` (`build_notebook.py:396`):
```python
LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
               class_weight="balanced", n_jobs=-1, random_state=SEED, verbose=-1)
```
- `num_leaves=31` — caps tree complexity (2^5 - 1; roughly analogous to
  `max_depth=5` for a balanced tree, but leaf-wise growth can make trees
  deeper in some branches).
- `class_weight="balanced"` — same imbalance handling as Random Forest /
  Logistic Regression.

**XGBoost vs. LightGBM, in one sentence:** both are gradient-boosted decision
trees with sequential error-correction; LightGBM's leaf-wise growth and
histogram binning usually make it faster and more memory-efficient on larger
datasets, while XGBoost's level-wise growth is often slightly more
conservative against overfitting on smaller datasets. In this repo, LightGBM
ends up being the model carried forward (`build_notebook.py:462`) based on
PR-AUC.

### 2.5 Why "pipelines"? `ColumnTransformer` & leakage

A **pipeline** chains preprocessing steps and a final estimator into a single
object with one `fit`/`predict` API. `fraud_utils.build_preprocessor()`
(`fraud_utils.py:338`) builds a `ColumnTransformer` that applies *different*
preprocessing to numeric vs. categorical columns:

```python
ColumnTransformer([
    ("num", Pipeline([imputer, optional StandardScaler]), numeric_features),
    ("cat", Pipeline([most-frequent imputer, OneHotEncoder]), categorical_features),
])
```

`FraudModelPipeline` (`fraud_utils.py:602`) then wraps
`Pipeline([("preprocess", ColumnTransformer), ("model", estimator)])`.

**Why this matters — data leakage.** If you fit a scaler or imputer on the
*entire* dataset before splitting into train/test, statistics from the test
set (its mean, its mode, etc.) "leak" into the training process — the model
gets a sneak peek at information it wouldn't have at prediction time, making
your evaluation **optimistic**. Putting preprocessing *inside* the pipeline
means `.fit()` only ever sees the training fold; `cross_val_score` and
`train_test_split`-style workflows automatically refit the preprocessing on
each training fold. This is the single most common subtle bug in ML
pipelines, and the deliberate `chargeback_reported` leakage trap in
`build_notebook.py` Section 4 is a more obvious version of the same family of
mistake (a feature that's only known *after* the outcome occurred).

---

## 3. Evaluation metrics: ROC-AUC, PR-AUC, precision, recall, F1

With only ~1.5% fraud, **accuracy is a trap**: a model that *always* predicts
"not fraud" is ~98.5% accurate while catching zero fraud. `evaluate_classifier()`
(`fraud_utils.py:387`) instead reports `roc_auc`, `pr_auc`, `precision`,
`recall`, and `f1`. Here's what each one means and when it's the right one to
look at.

### The confusion matrix (the foundation)

Every threshold-based metric below is built from four counts, given a chosen
**decision threshold** (default 0.5):

|                  | Predicted Fraud | Predicted Not Fraud |
|------------------|:---------------:|:--------------------:|
| **Actually Fraud**     | True Positive (TP)  | False Negative (FN) |
| **Actually Not Fraud** | False Positive (FP) | True Negative (TN) |

- **False Negative** = missed fraud (costly: the fraud goes through).
- **False Positive** = a legitimate transaction incorrectly flagged (costly:
  annoyed customer, manual-review workload).

### Precision, Recall, F1

- **Recall** (a.k.a. sensitivity, true positive rate) = `TP / (TP + FN)` —
  *of all the fraud that actually happened, what fraction did we catch?*
- **Precision** = `TP / (TP + FP)` — *of everything we flagged as fraud, what
  fraction was actually fraud?*
- There's an inherent **trade-off**: lowering the threshold flags more
  transactions, catching more fraud (↑ recall) but also flagging more
  legitimate transactions (↓ precision), and vice versa.
- **F1 score** = the harmonic mean of precision and recall:
  `F1 = 2 * (precision * recall) / (precision + recall)`. It's a single
  number that penalises models which are very lopsided (e.g. 99% precision
  but 1% recall gives a low F1, not a high average).

**Which matters more is a business decision** (`build_notebook.py:434`): an
auto-decline system needs high *precision* (don't anger good customers with
false declines); a "queue for human review" system can tolerate lower
precision in exchange for higher *recall* (catch more, let analysts filter).

### ROC-AUC

The **ROC curve** plots **True Positive Rate (recall)** vs. **False Positive
Rate** (`FP / (FP + TN)`) as the decision threshold sweeps from 1 to 0.
**ROC-AUC** is the area under that curve — the probability that the model
ranks a randomly-chosen positive example *above* a randomly-chosen negative
example. A value of 0.5 is random guessing; 1.0 is perfect ranking.

- ROC-AUC is **threshold-independent** — it summarises ranking quality across
  *all* thresholds at once.
- **Caveat with imbalanced data**: the false-positive rate's denominator
  (`FP + TN`) is dominated by the huge number of true negatives, so ROC-AUC
  can look deceptively high (close to 1.0) even when precision at any
  reasonable threshold is poor. This is why the notebook treats it as
  "useful but optimistic" (`build_notebook.py:433`) under heavy imbalance.

### PR-AUC (Average Precision)

The **precision-recall curve** plots precision vs. recall as the threshold
sweeps. **PR-AUC** (computed via `average_precision_score`, sometimes called
"Average Precision" or AP) is the area under that curve.

- Unlike ROC-AUC, PR-AUC has **no true negatives in its formula** —
  precision and recall are both about how well you find and correctly label
  the *positive* class. This makes it much more sensitive to performance on
  the rare class.
- The relevant baseline ("no-skill") for PR-AUC is the **prevalence of the
  positive class** (here, ~1.5%), *not* 0.5 — a random classifier gets
  PR-AUC ≈ prevalence, whereas it always gets ROC-AUC ≈ 0.5.
- This is why PR-AUC is the **primary model-selection metric** in this repo
  (`build_notebook.py:432`) — it's the honest view of "how good is this model
  at finding rare fraud," and it's what's used to rank models
  (`results_df.sort_values("pr_auc", ...)`, `build_notebook.py:423`) and to
  score cross-validation (`scoring="average_precision"`,
  `build_notebook.py:490`).

**Rule of thumb for interviews:** *"Use PR-AUC (and precision/recall) when
the positive class is rare and you mainly care about finding it; use ROC-AUC
when classes are roughly balanced or you care about ranking quality
generally."*

---

## 4. Cross-validation

A single train/test split gives you **one** estimate of performance, which
could be lucky or unlucky depending on which rows ended up in the test set.
**Cross-validation (CV)** repeats the train/evaluate process on multiple
different splits ("folds") and reports the mean and spread — a much more
reliable estimate, and the spread itself (`std`) tells you how *stable* the
estimate is.

`build_notebook.py` Section 2.4 compares two ways of constructing folds:

### Stratified k-fold

**K-fold CV** splits the data into `k` roughly equal chunks ("folds"). For
each fold in turn, that fold becomes the test set and the other `k-1` folds
become the training set — so every row is used for testing exactly once and
for training `k-1` times. **Stratified** k-fold additionally ensures each
fold preserves the overall class ratio (so every fold has ~1.5% fraud, not
some folds with 0% by chance) — important whenever you have class imbalance.

```python
StratifiedKFold(5, shuffle=True, random_state=SEED)
```

This is the standard choice for **i.i.d.** data (rows that don't have a
meaningful order). Its weakness here: with `shuffle=True`, a fold's training
set can contain transactions from *after* the test fold in time — i.e. the
model is allowed to "see the future" relative to what it's being tested on.

### TimeSeriesSplit

`TimeSeriesSplit(5)` creates folds that respect chronological order: fold 1
trains on the earliest chunk and tests on the next chunk; fold 2 trains on
everything up to that point and tests on the chunk after; and so on. **The
training set for every fold only contains data from *before* the test set.**

For a process that **drifts over time** (fraud patterns evolve as fraudsters
adapt), this is the realistic simulation of deployment — you always predict
the future from the past, never the reverse. `build_notebook.py:500` notes:
*"The time-series estimate is the number we would quote to the client...If it
were much lower than the stratified estimate, that gap would be a warning
that the model leans on patterns that do not persist over time."*

**Interview-ready summary:** *"Cross-validation gives a more robust estimate
of generalisation performance than a single split. For i.i.d. data, stratify
by the target to keep class balance consistent across folds. For
time-ordered/non-stationary data, use a time-respecting split
(`TimeSeriesSplit`) — otherwise you'll get an overly optimistic estimate from
training on 'future' data."*

---

## 5. Handling class imbalance (incl. SMOTE)

With ~1.5% fraud, a naive model has very little signal to learn the minority
class from, and a default 0.5 threshold will rarely predict "fraud" at all.
`build_notebook.py` Section 2.5 compares **three** standard remedies — it's
worth knowing all three and that they're not mutually exclusive.

### 1. Class weighting

Every model above can be told `class_weight="balanced"` (or, for XGBoost,
`scale_pos_weight=<neg_count/pos_count>`). This **re-weights the loss
function** so that getting a minority-class example wrong costs more than
getting a majority-class example wrong — *without changing the data at all*.
The model is mathematically nudged to pay more attention to fraud examples
during training.

- ✅ No data is duplicated or invented; cheap and easy to apply.
- ❌ Doesn't add any new *information* — the model still only sees the
  original fraud examples, just weighted more heavily.

### 2. SMOTE (Synthetic Minority Over-sampling Technique)

**The idea.** SMOTE creates **new, synthetic minority-class examples** by
picking a real minority example, finding its nearest minority-class
neighbours in feature space, and generating a new point somewhere *along the
line* between them (linear interpolation). The effect is to oversample the
minority class with plausible "in-between" points rather than just duplicating
existing rows.

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

smote = ImbPipeline([
    ("preprocess", fu.build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, scale=False)),
    ("smote", SMOTE(random_state=SEED)),
    ("model", LGBMClassifier(**base_lgbm)),
]).fit(X_train_feat, y_train)
```
(`build_notebook.py:529`)

**The critical detail — SMOTE only inside the pipeline, on training folds.**
Notice this uses `imblearn`'s `Pipeline`, not scikit-learn's. This matters
because **SMOTE must never see the test/validation data**:

- If you oversample *before* splitting, synthetic points derived from a
  test-set example (or its neighbours) could end up in the training set —
  the model would effectively be evaluated on data it was partly trained on,
  inflating performance.
- `imblearn.pipeline.Pipeline` makes `SMOTE` a pipeline *step* that only runs
  during `.fit()` (training), and is automatically skipped during
  `.predict()` / `.transform()` — so cross-validation correctly applies SMOTE
  fresh on each training fold and evaluates on untouched real data.

- ✅ Gives the model genuinely new (synthetic) examples to learn minority
  patterns from, rather than just reweighting existing ones.
- ❌ Synthetic points are interpolations — in regions where the minority
  class is sparse or the decision boundary is complex/non-linear, SMOTE can
  generate unrealistic points; can also amplify label noise if a minority
  example was mislabeled.

### 3. Threshold tuning

Train the model normally (probabilities), then **choose the decision
threshold deliberately** rather than defaulting to 0.5. `build_notebook.py:547`
sweeps thresholds from the precision-recall curve and picks the one that
maximises recall subject to precision ≥ 50% — i.e. *"never flag below 50%
precision, but otherwise catch as much fraud as possible."*

> **Interpretation from the notebook (`build_notebook.py:578`):** *"All three
> levers move the same precision/recall trade-off; they are not magic. ...
> threshold tuning is the cheapest, most controllable lever — it lets the
> business pick its operating point without retraining."* In practice,
> threshold tuning is usually the first lever pulled (free, instant,
> reversible), with class weighting/SMOTE used to improve the underlying
> probability estimates the threshold is applied to.

---

## 6. Model interpretability: permutation importance & SHAP

Once you have a model, the next question is usually *"why does it predict
what it predicts?"* — both to sanity-check it (is it using sensible signal?)
and to explain individual decisions to stakeholders (`build_notebook.py`
Section 2.6 and 5.1).

### Permutation importance

**The idea.** To measure how important a feature is, **shuffle (permute) its
values** across all rows — destroying any real relationship between that
feature and the target — and see how much the model's performance drops.
A feature the model relies on heavily will cause a big performance drop when
scrambled; an irrelevant feature won't change much.

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    fitted_pipes["LightGBM"], test_df.iloc[sample_idx][FEATURES], y_test.iloc[sample_idx],
    scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=-1,
)
```
(`build_notebook.py:597`)

- **Model-agnostic** — works for *any* fitted model, since it only needs
  `predict`/`predict_proba` and a scoring function.
- **Directly tied to predictive value** — the importance is measured in the
  units of whatever metric you choose (here, PR-AUC drop).
- Repeating the shuffle `n_repeats` times and averaging reduces noise from
  any single random shuffle.
- Caveat: if two features are highly correlated, permuting just one of them
  may not hurt performance much (the model can "fall back" on its correlated
  partner), which can understate the importance of correlated feature groups.

### SHAP (SHapley Additive exPlanations)

**The idea.** SHAP is based on **Shapley values** from cooperative game
theory: imagine each feature is a "player" contributing to the "payout" (the
model's prediction for one row), and you want to fairly split credit for that
prediction among the features, accounting for all the ways features interact.
The result is, for each individual prediction, a **per-feature contribution**
that:

- Is **additive**: `prediction = baseline (average prediction) + sum of each
  feature's SHAP value for this row`.
- Can be **positive or negative** — a feature can push the prediction *up*
  (toward "fraud") or *down* (toward "not fraud") for a specific row.

This gives two complementary views, both used in the notebook:

1. **Global view — the beeswarm plot** (`build_notebook.py:625`, using
   `shap.TreeExplainer` for the tree-based LightGBM model): one dot per
   (row, feature), showing the *distribution* of that feature's SHAP values
   across many transactions. This answers *"across the whole dataset, which
   features matter most, and in which direction?"* — e.g.
   `build_notebook.py:636`: *"foreign transactions, high-risk merchant
   categories, large amounts, and high transaction velocity push risk up,
   while long account tenure pulls it down."*

2. **Local view — the waterfall plot** (`build_notebook.py:1000`): for **one
   specific transaction**, shows exactly which features pushed the
   prediction up or down from the baseline, and by how much. This is what
   lets you write the plain-English explanation in
   `build_notebook.py:1008`: *"We flagged this transaction... because it was
   a foreign transaction, for a larger-than-typical amount, in a high-risk
   merchant category, on an account that hasn't been open long."*

**Permutation importance vs. SHAP, in one sentence:** permutation importance
tells you *how much a feature matters for overall performance* (global, by
shuffling); SHAP tells you *how much each feature contributed to each
individual prediction* (both global, by aggregating, and per-row). Agreement
between the two (as found here) is reassuring — it means the conclusions
aren't an artefact of one particular method.

**Why `TreeExplainer` specifically?** SHAP has several explainer
implementations; `TreeExplainer` exploits the structure of tree-based models
(like LightGBM) to compute *exact* Shapley values efficiently, rather than
needing the expensive model-agnostic approximations required for arbitrary
models (e.g. `KernelExplainer`).

---

## 7. Regularisation & hyperparameter tuning

Briefly, since these are covered in `build_notebook.py` Section 2.7:

- **Regularisation** adds a penalty for model complexity to the training
  objective, trading a little training-set fit for better generalisation:
  - **L2 (Ridge)** — penalises the sum of squared coefficients; shrinks all
    coefficients toward zero but rarely exactly to zero.
  - **L1 (Lasso)** — penalises the sum of absolute coefficients; can drive
    coefficients to **exactly zero**, performing automatic feature selection.
  - **ElasticNet** — a weighted blend of L1 and L2 (controlled by `l1_ratio`
    in scikit-learn's `LogisticRegression`).
- **Hyperparameter tuning with Optuna** — rather than exhaustively trying
  every combination of hyperparameters (grid search), Optuna uses **Bayesian
  optimisation**: it tries a configuration, observes the resulting score, and
  uses that information to choose the *next* configuration more cleverly —
  focusing the search on promising regions of the hyperparameter space. This
  finds strong configurations in far fewer trials than a grid search would
  need.

---

## 8. Fairness metrics

Covered in `build_notebook.py` Section 5.2 and implemented in
`fraud_utils.py` (lines 530–596). All three compare a **privileged** vs.
**unprivileged** group on the model's *predictions* (not the data itself):

- **Demographic parity difference** (`demographic_parity_difference`,
  `fraud_utils.py:530`) — the difference in *positive-prediction rates*
  (here, fraud-flag rates) between groups. Zero means equal flag rates.
- **Disparate impact ratio** (`disparate_impact_ratio`, `fraud_utils.py:546`)
  — the *ratio* of flag rates (unprivileged / privileged). The "four-fifths
  rule" (a U.S. legal guideline) treats ratios below 0.8 or above 1.25 as
  evidence of adverse impact.
- **Equalised odds difference** (`equalized_odds_difference`,
  `fraud_utils.py:567`) — separately compares the **true positive rate** and
  **false positive rate** between groups. This asks a different question than
  demographic parity: *not* "do we flag both groups equally often?" but "when
  the model is right (or wrong), is it right (or wrong) equally often for
  both groups?"

The key lesson from this repo (`build_notebook.py:1049`): **dropping a
sensitive attribute (e.g. gender) from the model's features is necessary but
not sufficient** — other features can act as **proxies** that are correlated
with the sensitive attribute, so the model can still end up treating groups
differently even without ever "seeing" the sensitive column directly.

---

## 9. Quick-reference glossary

| Term | One-line definition |
|---|---|
| **A/B test** | Randomised experiment comparing a control and treatment group to estimate a causal effect |
| **Power analysis** | Calculation of the sample size needed to detect a target effect with a target probability |
| **p-value** | P(data this extreme \| no real effect) — not the probability the effect is real |
| **Confidence interval** | A range of plausible values for a true effect size |
| **Logistic regression** | Linear model for binary outcomes via sigmoid(linear combination of features) |
| **Random Forest** | Ensemble of decision trees trained on bootstrapped rows/random feature subsets, averaged ("bagging") |
| **Gradient boosting** (XGBoost, LightGBM) | Ensemble of shallow trees trained sequentially, each correcting the previous ensemble's errors |
| **`ColumnTransformer`** | scikit-learn object applying different preprocessing to different columns |
| **Data leakage** | Information from outside the training data (e.g. the test set, or post-outcome facts) improperly influencing training/evaluation |
| **Confusion matrix** | 2×2 table of TP / FP / FN / TN at a given decision threshold |
| **Precision** | TP / (TP + FP) — of flagged items, how many were correct |
| **Recall (sensitivity)** | TP / (TP + FN) — of actual positives, how many were found |
| **F1 score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Probability a random positive is ranked above a random negative; baseline 0.5 |
| **PR-AUC (Average Precision)** | Area under precision-recall curve; baseline ≈ class prevalence; better for rare-class problems |
| **Cross-validation (CV)** | Repeated train/evaluate on different data splits ("folds") to get a robust performance estimate |
| **Stratified k-fold** | k-fold CV that preserves class proportions in every fold |
| **TimeSeriesSplit** | CV that respects chronological order — always trains on the past, tests on the future |
| **Class weighting** | Re-weighting the loss function to penalise minority-class errors more |
| **SMOTE** | Generates synthetic minority-class examples by interpolating between real minority neighbours |
| **Threshold tuning** | Choosing the probability cutoff for "positive" deliberately, instead of defaulting to 0.5 |
| **Permutation importance** | Feature importance measured by shuffling a feature and observing the performance drop |
| **SHAP value** | Per-feature, per-prediction contribution to a model's output, based on Shapley values from game theory |
| **Regularisation (L1/L2/ElasticNet)** | Penalty on model complexity/coefficient size to reduce overfitting |
| **Demographic parity** | Fairness metric: do groups get flagged at the same rate? |
| **Disparate impact ratio** | Fairness metric: ratio of flag rates between groups; <0.8 or >1.25 is a red flag |
| **Equalised odds** | Fairness metric: are TPR and FPR similar across groups? |
| **Model card** | One-page documentation summarising a model's purpose, data, metrics, and limitations |

---

## 10. Suggested reading order through the notebook

If you're working through `northwind_fraud_casestudy.ipynb` for the first
time, this is roughly the dependency order of ideas (matches the notebook's
own section order):

1. **Section 1 (Statistics)** — build intuition for hypothesis testing and
   uncertainty before touching ML; the A/B test result ("step-up auth cuts
   fraud but adds friction") motivates *why* Northwind also wants a
   real-time fraud model (Section 2).
2. **Section 2 (Supervised Learning)** — the core ML workflow: split → train
   four models in pipelines → evaluate with PR-AUC/ROC-AUC →
   cross-validate → handle imbalance → interpret with SHAP → tune.
   This is the densest section and the one to revisit most.
3. **Section 3 (Production Python)** — look at `fraud_utils.py` and
   `test_fraud_utils.py` alongside this to see how the same logic from
   Section 2 is written as tested, reusable code rather than notebook
   scratch.
4. **Section 4 (Data Wrangling)** — the `chargeback_reported` leakage trap
   here is the same *category* of bug as "fit your scaler on the test set" —
   seeing it in an obvious form first makes the subtler pipeline-leakage
   point in Section 2 click.
5. **Section 5 (Interpretability & Ethics)** — builds directly on the SHAP
   and model objects from Section 2; the fairness audit is the natural
   "what could go wrong after deployment" follow-up.
6. **Section 6 (Visualisation)** — can be read any time; it's about
   *communicating* the findings from the earlier sections.

For interview prep specifically: be ready to explain, from memory and without
the notebook open, *why* each metric/model/technique was chosen over the
alternatives — the "Interpretation" callouts throughout `build_notebook.py`
are written exactly with that framing.
