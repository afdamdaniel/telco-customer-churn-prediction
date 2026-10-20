# Telco Customer Churn Prediction

Predict which telecom customers are likely to leave. The project compares three classification models in a single Jupyter notebook.

## Overview

A telecom company loses money when customers leave. If we can spot them early, the company can try to keep them.

This notebook cleans the data, explores it, builds a preprocessing pipeline, and trains three models:

- Logistic Regression
- Balanced Random Forest
- XGBoost

## Dataset

- Source: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/abdallahwagih/telco-customer-churn)
- 7,043 customers and 33 columns. Each row is one customer.
- Target: `Churn Label` (Yes or No). 1,869 customers churned, which is 26.5%.
- Main columns: contract type, payment method, monthly charges, total charges, tenure, and the services each customer uses.

The data file `Telco_customer_churn.xlsx` is **not** in this repo. Download it from Kaggle and put it in the same folder as the notebook. Check the license on the Kaggle page before you reuse the data.

## What the notebook does

1. Loads the data and checks its shape, types, missing values and duplicates.
2. Converts `Total Charges` from text to numbers. This leaves 11 missing values, which are filled later.
3. Explores the data with churn counts, histograms, churn by contract type, and a correlation table.
4. Applies a log transform to `Tenure Months`, `Monthly Charges` and `Total Charges`.
5. Drops columns that are not useful or that reveal the answer: `CustomerID`, `Churn Label`, `Churn Value`, `Churn Score`, `Churn Reason`, `Zip Code`, `Latitude`, `Longitude`, `Gender` and `City`.
6. Splits the data 80/20 with a stratified split (`random_state=42`).
7. Builds a pipeline:
   - Numbers: fill missing values with the mean, then scale.
   - Categories: fill missing values with the most common value, then one-hot encode.
8. Trains the three models and compares them on the test set.

## Results

Test set: 1,409 customers (374 churned, 1,035 stayed).

| Model | Accuracy | Churn precision | Churn recall | Churn F1 |
|---|---|---|---|---|
| Logistic Regression (balanced class weights) | 0.754 | 0.53 | 0.76 | 0.62 |
| Balanced Random Forest | 0.769 | 0.55 | 0.72 | 0.62 |
| XGBoost | 0.786 | 0.61 | 0.53 | 0.57 |

Confusion matrix counts:

| Model | True negative | False positive | False negative | True positive |
|---|---|---|---|---|
| Logistic Regression | 780 | 255 | 91 | 283 |
| Balanced Random Forest | 815 | 220 | 105 | 269 |
| XGBoost | 911 | 124 | 177 | 197 |

A false positive is a customer who stayed but was predicted to leave. A false negative is a customer who left but was predicted to stay.

The XGBoost precision and F1 are worked out from its confusion matrix. The notebook only prints its accuracy.

### Takeaways

- Logistic Regression finds the most churners (283 of 374).
- Logistic Regression and Balanced Random Forest are almost tied
.
- XGBoost has the best accuracy and the fewest false alarms. But it misses 177 of the 374 churners.
- The best choice depends on cost. Is a missed churner worse than a wasted retention offer?


