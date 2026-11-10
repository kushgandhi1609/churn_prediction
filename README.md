# Customer Churn Prediction

An end-to-end machine learning project: data cleaning, exploratory analysis, model comparison, evaluation, and a Streamlit web app that estimates the chance a telecom customer will leave.

**Live app:** _add the link here after deployment_

This README is also a decision log. For every major step it records **what I did, why, and what the data showed**, including the places where the result was not what I expected.

---

## Table of contents
1. [Problem and goal](#1-problem-and-goal)
2. [Data](#2-data)
3. [Environment and project structure](#3-environment-and-project-structure)
4. [Data cleaning decisions](#4-data-cleaning-decisions)
5. [Exploratory data analysis](#5-exploratory-data-analysis)
6. [Train/test split](#6-traintest-split)
7. [Preprocessing pipeline](#7-preprocessing-pipeline)
8. [Model selection](#8-model-selection)
9. [Handling class imbalance](#9-handling-class-imbalance)
10. [Hyperparameter tuning](#10-hyperparameter-tuning)
11. [Choosing the decision threshold](#11-choosing-the-decision-threshold)
12. [Interpreting the model and the final feature set](#12-interpreting-the-model-and-the-final-feature-set)
13. [Final evaluation on the test set](#13-final-evaluation-on-the-test-set)
14. [Saving the model](#14-saving-the-model)
15. [The Streamlit app](#15-the-streamlit-app)
16. [Deployment](#16-deployment)
17. [Limitations and future work](#17-limitations-and-future-work)
18. [Run it yourself](#18-run-it-yourself)

---

## 1. Problem and goal

**Churn** means a customer leaves the company. Winning a new customer usually costs more than keeping an existing one, so a company benefits from knowing which customers are likely to leave *before* they do, and offering them a discount or a call.

- **Task:** binary classification. Target `Churn` (1 = left, 0 = stayed).
- **Output:** a churn probability for a customer, plus a high-risk / lower-risk flag.

**Why this topic:** it covers every stage of a real ML project (messy tabular data, class imbalance, feature engineering, metric choice, explainability, deployment), it has a clear business story, and it avoids the over-used beginner datasets (Titanic, Iris, MNIST).

## 2. Data

[Telco Customer Churn on Kaggle (uploaded by *blastchar*)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): file `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

- 7,043 customers, 21 columns.
- Class balance: 5,174 stayed (73.5%), 1,869 churned (26.5%).
- The raw file is **not** included in this repository. Download it from Kaggle and place it in `data/`.

**Decision: verify the dataset.** Several datasets on Kaggle share this name. I confirmed the right one by checking the shape (`(7043, 21)`), the class counts above, and the first column names.

**Decision: look at the target balance before anything else.** A 26.5% churn rate means a model that always predicts "stay" scores 73.5% accuracy while being useless. This is why accuracy is not used to judge models in this project.

## 3. Environment and project structure

- Python 3.14.7 in a virtual environment (`venv`) on macOS.
- scikit-learn 1.9.1, pandas 3.1.1, joblib 1.6.0, Streamlit 1.67.0 (pinned in `requirements.txt`).

```
app/app.py              Streamlit app
models/                 saved pipeline (preprocessing + model in one file)
notebooks/
  01_eda.ipynb          cleaning and exploration
  02_modeling.ipynb     split, pipeline, models, tuning, evaluation
requirements.txt
```

**Decisions**
- **Local Jupyter, not Google Colab.** The project ends in a deployed app, so it needs a real project folder with scripts, a saved model, a requirements file and Git. Colab sessions are temporary.
- **Two notebooks.** Each tells one story. EDA reads the data; modeling builds on it. Keeping them apart also prevents re-running one-time cleaning steps (such as mapping `Churn` to 0/1) on data that has already been cleaned.
- **Notebooks for exploration, `.py` files for what gets deployed.**
- **`requirements.txt` lists only the four libraries the app needs**, not the output of `pip freeze`, which would include JupyterLab and dozens of notebook-only packages. `scikit-learn` is pinned to the exact training version because a saved model is tied to the version that created it.

## 4. Data cleaning decisions

Each rule below is a fixed rule applied row by row. None of them learns anything from other rows, so applying them before the train/test split does not leak information.

| Issue found | Decision | Reason |
|---|---|---|
| `TotalCharges` stored as text; `isnull()` reported 0 missing | Inspected the blank rows, then converted to numbers | The blanks were spaces, not true nulls, so `isnull()` could not see them |
| 11 blank `TotalCharges` rows, all with `tenure = 0` and `Churn = No` | Filled with **0** | These are brand-new customers who have not been billed yet. Zero is the true value. Dropping them would lose real customers; filling with the mean would claim they had paid a typical lifetime amount |
| `Churn` is "Yes"/"No" text | Mapped to 1/0 | Models need numbers, and the mean of a 0/1 column is directly the churn rate |
| `customerID` | Dropped | Unique per row, so it carries no pattern and a model could only memorise it |
| "No internet service" (1,526 rows) and "No phone service" (682 rows) inside service columns | Merged into "No" | This information already lives in `InternetService` and `PhoneService`. Repeating it as a third category in seven columns adds nothing |
| `gender` | Dropped | Churn rate was 26.9% (female) vs 26.2% (male), so essentially no signal. It is also a protected attribute, which a deployed retention system should avoid using |

**Sanity checks after cleaning:** shape, missing values, data types, `Churn` values, and the Yes/No columns. No problems were found.

## 5. Exploratory data analysis

**Method:** for each categorical column, the churn rate per group (the mean of the 0/1 target). The overall rate is 26.5%, so a group far above or below that separates the outcomes.

| Feature | Group churn rates |
|---|---|
| Contract | Month-to-month **42.7%**, one year 11.3%, two year **2.8%** |
| Internet service | Fiber optic **41.9%**, DSL 19.0%, none 7.4% |
| Payment method | Electronic check **45.3%**, mailed check 19.1%, bank transfer (auto) 16.7%, credit card (auto) 15.2% |
| Tech support | No 31.2%, yes 15.2% |
| Online security | No 31.3%, yes 14.6% |
| Paperless billing | Yes 33.6%, no 16.3% |
| Senior citizen | Yes 41.7%, no 23.6% |
| Partner / dependents | No partner 33.0% vs 19.7%; no dependents 31.3% vs 15.5% |
| Gender | Female 26.9%, male 26.2% (no effect) |

**Numeric features (average by outcome)**

| | Stayed | Churned |
|---|---|---|
| Tenure (months) | 37.6 | 18.0 |
| Monthly charges | $61.3 | $74.4 |
| Total charges | $2,549.9 | $1,531.8 |

Correlation with churn: tenure -0.35, monthly charges +0.19, total charges -0.20.

**Charts:** new customers (first months of tenure) are the one place where churners outnumber stayers. Monthly charges show a large spike near $20 of loyal customers, mostly people with no internet service. Above about $70, churners make up a larger share of each bar.

**Lessons recorded while doing the EDA**
- **Counts tell you how big a group is; rates tell you how risky it is.** Reading bar heights in the contract chart suggests month-to-month customers mostly stay, because that group is the largest. The churn *rate* (about 43%) is the right measure.
- **Patterns are not causes.** Fiber customers churn more, but fiber is also more expensive, so price or competition may be the real reason.
- **Overlapping features.** `TotalCharges` has a 0.83 correlation with `tenure`, because total charges is roughly tenure times monthly price. This matters later for the linear model.
- No impossible values or outliers: tenure 0 to 72, monthly charges $18 to $119.

## 6. Train/test split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

- **80/20 split:** 5,634 training and 1,409 test customers.
- **`stratify=y`:** keeps the churn rate the same in both parts (26.54% in train, 26.54% in test). With an imbalanced target, a purely random split could leave the test set with noticeably more or fewer churners, which would make the final score partly luck.
- **`random_state=42`:** makes the split reproducible.
- **The test set stays locked.** All model choices were made with cross-validation on the training data. The test set was used only for final checks (see section 13).

## 7. Preprocessing pipeline

Done **after** the split, inside a scikit-learn `ColumnTransformer`:

- **`StandardScaler`** on numeric columns, so a column with large values (total charges, standard deviation about 2,267) does not dominate one with small values (tenure, about 25) in a linear model.
- **`OneHotEncoder(drop='if_binary', handle_unknown='ignore')`** on categorical columns. One-hot encoding avoids implying an order between categories (it would be wrong to number "Two year" as 3x "Month-to-month"). `drop='if_binary'` turns yes/no columns into a single 0/1 column. `handle_unknown='ignore'` means the deployed app will not crash on an unseen category.
- **`SeniorCitizen` goes through the same encoder** as the other two-value columns. It is already 0/1, so the result is identical and the code is simpler.

**Decision: why preprocessing comes after the split.** Scaling and encoding *learn from data* (means, spreads, category lists). If they were fit on all rows, information from the test customers would influence training. That is data leakage, and it makes test scores look better than they really are. The preprocessor is fit on the training data only (`fit_transform`) and then only applied to the test data (`transform`).

**Decision: use a `Pipeline`.** Putting the preprocessor and model in one object means (1) cross-validation re-fits the scaler and encoder inside each fold, so nothing leaks into validation folds, and (2) deployment needs a single saved object: raw customer in, prediction out.

## 8. Model selection

**Evaluation method:** 5-fold stratified cross-validation on the training set only (`shuffle=True, random_state=42`), so one lucky or unlucky split cannot mislead.

**Metrics, and why not accuracy:**

| Metric | Question it answers |
|---|---|
| Recall | Of customers who really churned, how many did we catch? |
| Precision | Of customers we flagged, how many really churned? |
| F1 | A single balance of precision and recall |
| ROC-AUC | How well does the model rank churners above stayers? (0.5 = coin flip) |
| Accuracy | Reported, but misleading here |

**Decision: start with a dummy baseline and logistic regression.** A baseline gives a score to beat. If complex models only barely improve on it, the complexity is not justified.

| Model (5-fold CV) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (always "stay") | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.803 | 0.655 | 0.545 | 0.595 | 0.846 |

The dummy model's 73.5% accuracy with zero recall shows why accuracy was not used to pick a model. Logistic regression had a good AUC but caught only about 55% of churners.

## 9. Handling class imbalance

**Decision: class weights first.** The model defaults to "stay" because that is usually right, so it only flags a customer when it is more than 50% sure. `class_weight='balanced'` makes each churner count about 2.8 times as much as a stayer during training. It is a one-line change, with no synthetic data to explain.

| Model (5-fold CV) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.803 | 0.655 | 0.545 | 0.595 | 0.846 |
| Logistic regression, balanced | 0.750 | 0.519 | **0.804** | 0.631 | 0.846 |
| Random forest, balanced | 0.780 | 0.575 | 0.656 | 0.613 | 0.827 |
| XGBoost (`scale_pos_weight` about 2.8) | 0.753 | 0.522 | 0.798 | 0.631 | 0.845 |

**What this showed**
- Recall rose from 55% to 80%, at the cost of precision (66% to 52%) and accuracy (80% to 75%). That was a deliberate trade: for a retention team, **missing a churner (a lost customer) costs more than a false alarm (an unnecessary offer)**.
- **AUC did not change** (0.846 both times). Class weights did not make the model better at ranking customers; they moved the point where it starts flagging. Real improvement has to show up in AUC.
- **The simplest model matched the most complex.** Logistic regression and XGBoost are tied (the 0.001 AUC gap is noise), and random forest was weakest. The signal in this dataset is mostly simple and additive (contract, tenure, internet type), and with about 5,600 training rows tree models have little room to add anything.

## 10. Hyperparameter tuning

- Logistic regression: `GridSearchCV` over `C` in {0.01, 0.1, 1, 10}, scored on ROC-AUC. Best `C = 10` (CV AUC 0.8463), meaning almost no regularization was needed.
- XGBoost: `RandomizedSearchCV`, 20 random combinations of trees, depth, learning rate and subsample. Best: 100 trees, depth 2, learning rate 0.1, subsample 0.7 (CV AUC 0.8489).

| Model | Default AUC | Tuned AUC |
|---|---|---|
| Logistic regression | 0.846 | 0.846 |
| XGBoost | 0.845 | 0.849 |

**Decision: choose logistic regression.** The 0.004 XGBoost gain is within noise (and a best-of-many-searches score is slightly optimistic). XGBoost also picked the smallest depth on offer, a sign that the data wants a simple additive model. Logistic regression is easier to explain, faster, and simpler to deploy.

## 11. Choosing the decision threshold

The model outputs a probability; a threshold decides who is flagged. To choose it without touching the test set, I used **out-of-fold predictions on the training data** (each customer's probability comes from a model that never saw that customer).

| Threshold | Precision | Recall | F1 | Customers flagged |
|---|---|---|---|---|
| 0.30 | 0.424 | 0.926 | 0.582 | 57.9% |
| 0.40 | 0.471 | 0.868 | 0.610 | 49.0% |
| 0.50 | 0.517 | 0.801 | 0.629 | 41.1% |
| 0.60 | 0.571 | 0.718 | 0.636 | 33.4% |
| 0.70 | 0.629 | 0.591 | 0.610 | 24.9% |

**Decision: 0.50.**
- Recall stays around 80%, so about four in five churners are caught.
- It flags about 41% of customers, a manageable list, and a flagged customer is about twice as likely to churn as a random one.
- F1 hardly changes between 0.45 and 0.65, so the choice is not fragile.
- F1 peaks at 0.60, but F1 treats precision and recall as equal, and here missing a churner is worse than a false alarm.
- Going from 0.60 to 0.50 catches about 22 more churners per 1,000 customers but adds about 55 more false alarms (about 2.5 false alarms per extra churner caught). A business with expensive retention offers could reasonably raise the threshold. The right value depends on real offer costs, which this dataset does not contain.

## 12. Interpreting the model and the final feature set

**First look at the coefficients (all features).** The strongest effects matched the EDA: fiber optic and month-to-month contracts push toward churn; no internet, two-year contracts and long tenure push away. But some signs looked wrong. `MonthlyCharges` had a coefficient of -1.78 (higher bills *reducing* churn), while churners actually paid more on average.

**Why:** a linear model's coefficient is a feature's effect *with everything else held fixed*. `MonthlyCharges` is roughly the sum of the prices of the services a customer has, and `TotalCharges` is roughly tenure times monthly price. When columns overlap, the model splits their effects in odd ways that cancel out. This does not hurt predictions, but individual coefficients become unreliable to read.

**Experiments (5-fold CV on the training set, test set untouched)**

| Features | ROC-AUC | Recall | Coefficients |
|---|---|---|---|
| tenure + MonthlyCharges + TotalCharges | 0.846 | 0.804 | messy |
| tenure + TotalCharges | 0.846 | 0.804 | mostly sensible |
| tenure + MonthlyCharges | 0.845 | 0.793 | `MonthlyCharges` still negative |

My prediction that dropping `TotalCharges` would make the `MonthlyCharges` coefficient positive was **wrong**: services such as fiber and streaming already carry the "pays more" effect, so `MonthlyCharges` pulls back against them.

**Decision: final features = `tenure` + `MonthlyCharges` + the 15 categorical columns (17 features).** Scores were effectively tied, so I chose on usability: a person entering a customer knows "24 months" and "$70 per month", but nobody knows `TotalCharges` offhand, and computing it from the other two would be an approximation the model was never trained on. The cost is a less tidy coefficient list, so the "what drives churn" story in this README relies on the **EDA churn rates**, not on individual coefficients for charges or streaming.

## 13. Final evaluation on the test set

The test set (1,409 customers the model never saw) was checked for **two** model versions: the first model (with `TotalCharges`) and the final one. The final choice was made from cross-validation, not from these results.

| | First model | **Final model** |
|---|---|---|
| ROC-AUC | 0.841 | **0.838** |
| Recall (churners) | 0.79 | **0.78** |
| Precision (churners) | 0.50 | **0.51** |
| Customers flagged | 583 (41%) | 571 (40.5%) |

**Final model confusion matrix (threshold 0.50)**

|  | Predicted stayed | Predicted churned |
|---|---|---|
| **Actually stayed** | 754 | 281 (false alarms) |
| **Actually churned** | 84 (missed) | 290 (caught) |

- Catches **290 of 374** churners (78%); about 1 in 5 still slips through.
- About half of the flagged customers really churn, compared with a 26.5% base rate.
- Overall accuracy is 0.74. It looks mediocre, but the dummy model also scores 0.735 while catching no churners at all.
- Test scores are close to cross-validation (AUC 0.845, recall 0.793), so the model generalises and there is no sign of leakage.

## 14. Saving the model

The whole pipeline (preprocessing + model) is saved with `joblib` to `models/churn_model.joblib`. It expects the 17 raw columns, with the same names and text values as in training.

**Check before building the app:** I reloaded the file and predicted two made-up customers.
- High risk (2 months tenure, fiber, month-to-month, electronic check, no add-ons, $80): **0.78**
- Low risk (60 months, two-year contract, DSL, automatic card payment, security and support on, $55): **0.03**

The probabilities move in the right direction for the features that drove churn in the EDA.

## 15. The Streamlit app

`app/app.py` is a form that builds a one-row table in the same format as the training data and returns the churn probability and a high-risk / lower-risk message (threshold 0.50).

**Decisions**
- **Streamlit** turns a Python script into a web page without any HTML or JavaScript, and deploys for free.
- **Business rules in the form.** Training merged "No internet service" into "No", so if the user picks *Internet service = No*, the six internet add-on fields are set to "No" automatically. If *Phone service = No*, *Multiple lines* is set to "No".
- **Input ranges match the data** (tenure 0 to 72 months, monthly charges $18 to $120).
- **The model is loaded once** (`st.cache_resource`) and found relative to the script, so it works from any folder.
- **Verified against the notebook:** the same customer gives the same probability in both (about 78% for the high-risk example).

**Known quirk:** with the other fields fixed, lowering monthly charges can *raise* the predicted risk (a $70 bill gave 86% where $80 gave 78%). This comes from the overlap between charges and service columns described in section 12, not from a bug. Effects of changing monthly charges alone should not be read as causal.

## 16. Deployment

- Code on GitHub; the app is deployed on Streamlit Community Cloud from `app/app.py` on the `main` branch.
- `requirements.txt` pins the exact versions used for training.
- The Python version on Streamlit Community Cloud is chosen in the app's *Advanced settings*; a `runtime.txt` file is not used for this. I trained on Python 3.14, so if a package fails to install on the hosting Python version, the fix is to pick a different version in those settings and redeploy.
- `.gitignore` excludes `venv/`, notebook checkpoints and the raw dataset. `models/churn_model.joblib` is **committed** because the deployed app needs it.

## 17. Limitations and future work

**Limitations**
- Charge-related coefficients overlap with service columns, so single coefficients (especially for charges and streaming) should not be interpreted separately or causally.
- One telecom dataset; results may not transfer to other companies or periods.
- The 0.50 threshold is a judgment call. It should be set from the real cost of a retention offer versus a lost customer.
- Precision is about 50%: roughly half of flagged customers would not have left.
- The test set was examined for two model versions. The final choice was driven by cross-validation, but this is worth stating.

**Future work**
- Train with and without `gender` and compare cross-validation scores, to confirm that dropping it costs nothing.
- Add SHAP explanations to show *why* an individual customer is flagged.
- Try SMOTE or other resampling and compare with class weights.
- Add experiment tracking (for example MLflow) and a Docker container.
- Calibrate probabilities, so "72%" can be read as a real likelihood.

## 18. Run it yourself

```bash
git clone <this-repo-url>
cd churn-prediction
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

To reproduce the analysis, also install `jupyterlab`, `matplotlib`, `seaborn`, `xgboost` and `imbalanced-learn`, download the dataset into `data/`, and run `notebooks/01_eda.ipynb` followed by `notebooks/02_modeling.ipynb`.
