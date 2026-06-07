# Employee Attrition Prediction

A machine learning project that predicts employee churn using the IBM HR Analytics dataset. The notebook walks through end-to-end binary classification — from EDA and preprocessing to logistic regression with threshold tuning and feature importance analysis.

---

## Overview

Employee attrition is costly. This project builds a classifier that flags employees likely to leave, prioritising **recall** over raw accuracy to minimise missed at-risk cases in an imbalanced dataset.

**Dataset:** [IBM HR Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (`WA_Fn-UseC_-HR-Employee-Attrition.csv`)

---

## Workflow

1. **Data Loading & Inspection** — Load the CSV, inspect head, shape, and data types
2. **Feature Pruning** — Drop zero-variance columns (`Over18`, `EmployeeCount`, `StandardHours`) and ID columns (`EmployeeNumber`)
3. **Target Encoding** — Map `Attrition` → `{Yes: 1, No: 0}`
4. **Class Distribution** — Pie chart showing the ~84/16 class imbalance
5. **Feature Selection** — Drop low-importance features identified from coefficients (`Department`, `EducationField`, `MaritalStatus`, `HourlyRate`, `JobLevel`, `PerformanceRating`, `MonthlyRate`, `DailyRate`)
6. **Preprocessing** — One-hot encode categoricals with train/test alignment; `StandardScaler` on numericals
7. **Modelling** — Logistic Regression (`max_iter=1000`)
8. **Threshold Tuning** — Custom probability threshold (`0.15`) to boost recall for the minority class
9. **Evaluation** — `classification_report` (precision, recall, F1)
10. **Feature Importance** — Bar chart of logistic regression coefficients

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing, modelling, evaluation |

---

## Key Design Decisions

**Why recall over accuracy?**
The dataset is heavily imbalanced (~84% stay, ~16% leave). A naive model that always predicts "stay" gets 84% accuracy but catches zero churners. Precision-recall tradeoff is evaluated via `classification_report`, and the decision threshold is lowered to 0.15 to catch more true positives at the cost of some false positives.

**Why drop those features?**
A first pass with all features generates a coefficient plot. Features with near-zero coefficients (and domain-justified low signal) are removed in a second pass, reducing noise without meaningful accuracy loss.

**Why encode after splitting?**
One-hot encoding is applied after `train_test_split`, then test columns are realigned to match train via `.reindex()`. This prevents data leakage from test-set categories influencing the encoding schema.

---

## Usage

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Place the dataset in the project root
# Download from: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

# Run the notebook
jupyter notebook churn.ipynb
```

---

## Results

The model is evaluated using precision, recall, and F1-score on the held-out test set. With the custom threshold, recall for the attrition class (`1`) improves significantly compared to the default 0.5 threshold, at an expected tradeoff in precision.

> See the `classification_report` output and feature importance plot in the notebook for full metrics.

---

## File Structure

```
.
├── churn.ipynb                          # Main notebook
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Dataset (not tracked in git)
└── README.md
```

---

## Future Improvements

- Try ensemble models (Random Forest, XGBoost) for comparison
- Use SMOTE or class weighting to handle imbalance more formally
- Add cross-validation instead of a single train/test split
- Build a simple inference script for single-employee prediction
