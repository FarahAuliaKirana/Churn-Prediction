# Repeat-Purchase Prediction & Experiment Design — Olist E-Commerce

[![CI](https://github.com/FarahAuliaKirana/Churn-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/FarahAuliaKirana/Churn-Prediction/actions/workflows/ci.yml)
· Python · R · scikit-learn · XGBoost · SHAP · FastAPI · Docker · GitHub Actions

🇮🇩 Versi Bahasa Indonesia (lebih detail): [README.id.md](README.id.md)

An end-to-end customer analytics project on the public Olist Brazilian e-commerce dataset: from a churn model that *looked* good, to finding out why it wasn't, to a redesigned repeat-purchase model validated out-of-time, and an experiment plan for the business action it suggests.

---

## Executive summary

- **The business problem is activation, not retention.** 96.9% of customers buy once. Only **1.25%** of new customers place a second order within 90 days.
- **The original churn model (ROC-AUC 0.82) was mostly measuring *when* customers bought, not *whether* they would return.** The churn label was equivalent to "last order before 19 Jul 2018". A placebo test showed the same features predict arbitrary purchase-date windows with AUC 0.68–0.80.
- **I rebuilt the problem with a snapshot design** (features before date T, label = new order in the 90 days after T) and validated it **out-of-time across 3 periods**.
- **Result: a real but modest signal.** A 4-feature RFM logistic regression ranks customers with a stable **6.1–6.7× lift in the top 1%**. It beats a recency-only baseline in every period, and **XGBoost never beats it**, so the simpler model is recommended.
- **Experiment plan (in R):** detecting a +20% uplift from a second-purchase voucher needs **~68K new customers (~11 months)**, and a 15% voucher only breaks even if it **doubles** repeat purchases at a 30% margin. Recommendation: test smaller or pay-on-return incentives, and only commit to effects large enough to detect.

---

## How the project evolved

| Step | What happened | Evidence |
|---|---|---|
| 1. Baseline churn model | XGBoost churn classifier, ROC-AUC 0.97 | NB04 |
| 2. Leakage #1 | `first_order_month` encoded the dataset cut-off; removing it dropped AUC to ~0.80 | NB04 |
| 3. Bug fixes | Inverted `scale_pos_weight`, threshold tuned on the test set, double-counted payments and reviews | NB03–NB04, `src/` |
| 4. Leakage #2 (placebo test) | Remaining features still mostly identified purchase timing | NB04 |
| 5. Redesign | Snapshot-based repeat-purchase target, out-of-time test, recency & RFM baselines, bootstrap significance | NB06 |
| 6. Backtest | 3 out-of-time periods; RFM logistic regression is the most stable model | NB06, `src/models/backtest.py` |
| 7. Experiment design | Survival, power, and break-even analysis for a second-purchase voucher | `r/ab_test_design.md` |

---

## Key results

### Repeat-purchase model (v2) — backtest across 3 out-of-time periods

Each fold re-runs hyperparameter selection, refit, and evaluation using only data available before its test period.

| Test snapshot | Repeaters | Recency-only (PR-AUC · top-1% lift) | **RFM logistic regression** | XGBoost |
|---|---|---|---|---|
| Dec 2017 | 312 | 0.0102 · 2.25× | **0.0179 · 6.10×** | 0.0158 · 4.81× |
| Mar 2018 | 403 | 0.0098 · 1.74× | **0.0249 · 6.70×** | 0.0207 · 5.96× |
| Jun 2018 | 440 | 0.0080 · 1.36× | 0.0188 · 6.36× | **0.0192 · 6.82×** |
| **Mean** | | 1.78× | **6.39×** (PR-AUC 3.0× base rate) | 5.86× (PR-AUC 2.8× base rate) |

- XGBoost beats the recency-only baseline in **3/3** periods (paired bootstrap, 95% CI above 0), so there is signal beyond purchase timing.
- XGBoost **never** significantly beats RFM logistic regression, which is higher in 2/3 periods and more stable.
- Strongest signals: a previous repeat purchase (3.4× base rate), recent purchase, and a good review on the first order. Shipping cost has little effect, contrary to the original hypothesis.

### Experiment design (R) — [`r/ab_test_design.md`](r/ab_test_design.md)

| Question | Answer |
|---|---|
| 90-day repeat rate of new customers | **1.25%** (Kaplan–Meier; 31% of "second orders" were same-day split orders and are excluded) |
| When do customers come back? | Mostly within the first 30 days → send the incentive in week 2 |
| Who returns faster? (Cox model) | São Paulo customers (HR 1.18), first review 4–5 (HR 1.25), more installments |
| Sample size to detect +20% uplift | **67,768 customers** (~10.7 months of new-customer enrolment) |
| Minimum detectable effect with 3 months | **+39%** relative |
| When does a 15% voucher break even? | Never if margin ≤ 15%; needs ≥ +100% uplift at a 30% margin |

---

## Repository map

```
notebooks/
  01_data_loading.ipynb          Load & merge 7 Olist tables
  02_eda.ipynb                   EDA, churn definition, confounding checks
  03_feature_engineering.ipynb   Customer-level features
  04_modeling.ipynb              v1 churn model, leakage investigation, placebo test
  05_final_report.ipynb          Executive report (Indonesian)
  06_snapshot_redesign.ipynb     v2 snapshot model, out-of-time test, backtest
r/
  ab_test_design.Rmd / .md       Survival + power + break-even analysis (R)
src/
  data/load.py                   Data loading & validation
  features/engineer.py           v1 features
  features/snapshot.py           v2 snapshot builder (features < T, label in [T, T+90d))
  models/train.py, evaluate.py   v1 training, threshold, lift
  models/snapshot_model.py       v2 models, ranking metrics, bootstrap
  models/backtest.py             v2 expanding-window backtest
  api/                           FastAPI service (serves the v1 model)
  monitoring/drift.py            Prediction logging & drift checks
tests/                           150 unit tests (leakage checks, metrics, API)
train_pipeline.py                v1 training CLI
train_snapshot_pipeline.py       v2 training CLI (--backtest for the 3-period backtest)
```

---

## How to run

```bash
git clone https://github.com/FarahAuliaKirana/Churn-Prediction.git
cd Churn-Prediction
pip install -r requirements.txt
```

Download the [Olist dataset from Kaggle](https://www.kaggle.com/olistbr/brazilian-ecommerce) into `data/raw/`, then:

```bash
python train_pipeline.py                      # v1 churn model (used by the API)
python train_snapshot_pipeline.py --backtest  # v2 repeat-purchase model + 3-period backtest
Rscript r/render.R                            # experiment design report (needs pwr, knitr, markdown)
uvicorn src.api.main:app --reload             # API docs at http://localhost:8000/docs
pytest tests/                                 # 150 tests
```

---

## Limitations & next steps

- **Weak signal, small positive class:** 312–618 repeaters per split, so bootstrap intervals are wide.
- **Prediction ≠ causation:** the model finds customers likely to return on their own. Deciding who should receive a voucher needs the A/B test, then uplift modeling.
- **The API still serves the v1 model.** Migrating to the v2 RFM logistic regression requires a new request schema (`recency_days`, `tenure_days`, ...).
- **Short history:** the backtest is limited to 3 periods because order data effectively stops at the end of August 2018.

---

## Dataset

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0). ~100K orders and 96K customers. **Data period:** September 2016 – October 2018 (orders effectively stop at the end of August 2018). This is historical data released by Olist; the project itself was built in 2026.

*Farah Aulia Kirana · 2026*
