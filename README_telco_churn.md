# 📉 Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6B23?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

An end-to-end machine learning notebook for **predicting telecommunications customer churn**, with an emphasis on **recall, class imbalance, and reproducible preprocessing**.

📓 **[Explore the complete notebook](Churn.ipynb)**

## 🎯 Business Problem

Predicting which customers might leave a telecom provider can help retention teams prioritize outreach. Since missing an actual churner may be costly, this project compares overall predictive accuracy with **precision, recall, and F1-score for the churn class**.

## 📊 Dataset

- **Source:** [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/abdallahwagih/telco-customer-churn)
- **Size:** 7,043 customers × 33 original columns
- **Target:** `Churn Label` (encoded as `No = 0`, `Yes = 1`)
- **Churned:** 1,869 customers (26.54%)
- **Stayed:** 5,174 customers (73.46%)
- **Example features:** Contract, Payment Method, Tenure Months, Monthly Charges, Total Charges, and subscribed services

The dataset is **not included** in the repository. Download `Telco_customer_churn.xlsx` from Kaggle and place it alongside `Churn.ipynb`. Check the dataset license before reusing it.

## 🔎 Workflow

1. **Data inspection:** Examine dimensions, data types, missing values, duplicates, and class distribution.
2. **Cleaning:** Convert `Total Charges` to numeric (creating 11 missing entries).
3. **EDA:** Explore churn distribution, numerical distributions, contract-related churn, and numerical correlations.
4. **Feature preparation:** Encode the churn target, apply `np.log(x + 1)` to `Tenure Months`, `Monthly Charges`, and `Total Charges`; remove the target and selected identifiers/leakage-related variables.
5. **Train/test split:** Stratified 80/20 split (`random_state=42`).
6. **Model pipelines:** Use `ColumnTransformer` and `make_pipeline` to preprocess and train three classifiers.
7. **Evaluation:** Compare test-set confusion matrices and churn-class metrics.

### ⚙️ Preprocessing pipeline

| Feature type | Pipeline step | Purpose |
|---|---|---|
| Numeric | `SimpleImputer(strategy="mean")` | Fill missing numerical values |
| Numeric | `StandardScaler()` | Standardize numerical features |
| Categorical | `SimpleImputer(strategy="most_frequent")` | Fill missing categories |
| Categorical | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` | Encode categories and handle unseen values |

```text
Input features
   ├── Numeric → Mean imputation → StandardScaler ─┐
   └── Categorical → Mode imputation → OneHotEncoder ─┤
                                                      ↓
                                              ColumnTransformer
                                                      ↓
                                              Classification model
```

The imputer, encoder, and scaler are fitted inside each model pipeline on training data. The notebook currently performs the fixed logarithmic transformations before splitting.

### 🤖 Models

| Model | Notebook configuration | Motivation |
|---|---|---|
| Logistic Regression | `class_weight="balanced", max_iter=1000` | Linear baseline with class weighting |
| Balanced Random Forest | `random_state=42` | Ensemble with imbalance-aware sampling |
| XGBoost | `random_state=42` | Gradient-boosted tree classifier |

## 📈 Results

**Test set:** 1,409 customers (374 churned, 1,035 stayed). Metrics below are **for the positive churn class** unless otherwise stated, calculated from the notebook's confusion matrices.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 75.44% | 52.60% | **75.67%** | 62.06% |
| Balanced Random Forest | 76.93% | 55.01% | 71.93% | **62.34%** |
| XGBoost | **78.64%** | **61.37%** | 52.67% | 56.69% |

### Confusion matrices

| Model | True negatives | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|
| Logistic Regression | 780 | 255 | **91** | 283 |
| Balanced Random Forest | 815 | 220 | 105 | 269 |
| XGBoost | **911** | **124** | 177 | 197 |

A **false negative** is a customer who churned but was predicted to stay. A **false positive** is a customer who stayed but was predicted to churn.

### 💡 Key findings

- **Highest churn recall: Logistic Regression (75.67%).** It finds 283 of 374 churners, missing the fewest actual churners.
- **Highest churn F1-score: Balanced Random Forest (62.34%).** Its advantage over Logistic Regression (62.06%) is small.
- **Highest accuracy and churn precision: XGBoost (78.64% and 61.37%).** It has fewer false positives, but misses 177 churners.

**Model selection depends on business cost.** A retention team that prioritizes finding churners may prefer Logistic Regression. Balanced Random Forest is a reasonable F1-oriented candidate, while XGBoost reduces unnecessary churn alerts. These comparisons use one test split and are not evidence that any model will work equally well in deployment.

## ▶️ How to run

```bash
git clone https://github.com/afdamdaniel/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
python -m pip install numpy pandas matplotlib seaborn openpyxl scikit-learn imbalanced-learn xgboost jupyter
jupyter notebook Churn.ipynb
```

Before running the notebook, place **`Telco_customer_churn.xlsx`** in the repository's root directory. The repository does not currently pin dependency versions, so results may differ across environments.

## 📁 Repository structure

```text
telco-customer-churn-prediction/
├── Churn.ipynb                 # EDA, preprocessing, training, evaluation
├── README.md                   # Project documentation
└── Telco_customer_churn.xlsx   # Download separately; not committed
```

## 🚧 Limitations and next steps

- **Check remaining leakage risks:** The current feature set still includes `Lat Long` and `CLTV`; review whether these fields are appropriate and available at prediction time. `Count`, `Country`, and `State` are also low-information or constant fields in this dataset.
- **Fix the notebook churn-rate cell:** Since `Churn Label` has already been mapped to integers, use `data["Churn Label"].mean() * 100`.
- **Correct confusion-matrix commentary:** Some notebook descriptions swap false positives and false negatives; the tables above use the actual matrix counts.
- **Avoid repeated splits** and consider moving all data-dependent preprocessing inside the training pipeline.
- **Add cross-validation and threshold tuning**, plus ROC-AUC and precision-recall curves.
- **Inspect train/test gaps:** XGBoost shows substantially higher training accuracy than test accuracy, warranting overfitting checks.
- **Add figures and pinned dependencies** when available; optionally include interpretable feature importance and a simple deployment demo.

## 📚 Acknowledgments

Dataset: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/abdallahwagih/telco-customer-churn).

Built as a hands-on machine learning project using Python, Scikit-learn, imbalanced-learn, and XGBoost.
