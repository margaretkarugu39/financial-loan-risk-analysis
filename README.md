# Machine Learning Model for Loan Approval

## FinTech Innovations — Data Science Project

> **Objective:** Develop a machine learning classification system that supports faster, more consistent, and data-driven loan approval decisions while considering predictive performance and financial risk.

---

## 📌 Project Overview

FinTech Innovations currently uses a manual loan approval process that can be time-consuming and may lead to inconsistent decisions.

This project applies machine learning to historical loan application data to predict whether an application will be approved. The analysis follows the CRISP-DM framework:

**Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation & Conclusion**

The project evaluates multiple classification algorithms, applies cross-validation and hyperparameter tuning, and incorporates a custom business-cost metric to reflect the financial consequences of classification errors.

---

## 🎯 Business Problem

The business faces two important types of prediction errors:

- **False Negative:** A potentially creditworthy applicant is predicted as not approved.
  - Estimated opportunity cost: **$8,000 per case**

- **False Positive:** An application is predicted as approved but belongs to the non-approved class.
  - Estimated financial loss: **$50,000 per case**

Because these costs are different, model evaluation cannot rely on accuracy alone.

### Business Cost Formula

**Total Business Cost = (False Negatives × $8,000) + (False Positives × $50,000)**

---

## 📊 Dataset

The dataset contains **20,000 historical loan applications** and **35 variables** covering applicant financial information, credit history, behavioral characteristics, and loan details.

### Target Variable

`LoanApproved`

- `0` = Not Approved
- `1` = Approved

### Data Quality Considerations

The analysis identified:

- Missing values in selected categorical and financial variables
- Currency-formatted values in `AnnualIncome`
- Outliers in several financial variables
- Class imbalance in the target
- Strong correlations among some numerical predictors

---

## 🔎 Exploratory Data Analysis

The exploratory analysis examined:

- Dataset structure and data types
- Numerical distributions
- Categorical distributions
- Target-class distribution
- Missing-value patterns
- Feature correlations
- Outliers
- Approval rates across categorical groups
- Relationships between important numerical variables
- Pairwise relationships between selected features

Visualizations were used throughout the analysis to support interpretation and identify data-quality and modeling considerations.

---

## ⚙️ Data Preparation

A reproducible preprocessing framework was created using:

- `Pipeline`
- `ColumnTransformer`
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `OrdinalEncoder`
- `FunctionTransformer`

### Preprocessing strategy

**Numerical features**
- Median imputation
- Standardization

**Annual Income**
- Currency symbols and commas removed
- Converted to numerical values
- Median imputation
- Standardization

**Ordinal Education Level**
- Missing-value imputation
- Ordered encoding from High School through Doctorate

**Categorical features**
- Most-frequent imputation
- One-hot encoding
- Unknown categories handled safely

The preprocessing was kept inside the modeling pipelines to reduce the risk of data leakage.

---

## 🤖 Models Evaluated

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. K-Nearest Neighbors (KNN)

The models were evaluated using:

- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion Matrix
- Estimated Business Cost

---

## 🔁 Cross-Validation and Hyperparameter Tuning

A **5-fold Stratified Cross-Validation** strategy was used to account for the imbalanced target distribution.

Hyperparameter tuning was performed using **GridSearchCV** for:

- Logistic Regression
- Random Forest
- Gradient Boosting

### Best tuned configurations

| Model | Key Parameters | Cross-Validated ROC-AUC |
|---|---|---:|
| Logistic Regression | C = 100, solver = lbfgs | 0.994730 |
| Random Forest | 200 trees, max depth = 20 | 0.977166 |
| Gradient Boosting | 200 estimators, learning rate = 0.1, max depth = 5 | 0.990073 |

---

## 📈 Final Test-Set Results

| Model | Accuracy | Precision | Recall | ROC-AUC | False Negatives | False Positives | Business Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tuned Logistic Regression | 0.9658 | 0.9297 | 0.9268 | 0.9945 | 70 | 67 | $3,910,000 |
| Tuned Random Forest | 0.9338 | 0.9003 | 0.8128 | 0.9789 | 179 | 86 | $5,732,000 |
| Tuned Gradient Boosting | 0.9598 | 0.9251 | 0.9048 | 0.9909 | 91 | 70 | $4,228,000 |

The final evaluation considers both statistical performance and the financial consequences of classification errors.

---

## 💰 Business Impact

The business-cost analysis demonstrates why financial consequences should be considered alongside conventional classification metrics.

For the evaluated tuned models:

- Tuned Logistic Regression produced **70 false negatives** and **67 false positives**, with an estimated business cost of **$3.91 million**.
- Tuned Gradient Boosting produced **91 false negatives** and **70 false positives**, with an estimated business cost of **$4.228 million**.
- Tuned Random Forest produced **179 false negatives** and **86 false positives**, with an estimated business cost of **$5.732 million**.

These values are estimates based on the business costs provided for the project.

---

## 🔍 Model Interpretation

The Logistic Regression model was examined through its coefficients to improve interpretability.

Variables with relatively large coefficient magnitudes included:

- Total Debt-to-Income Ratio
- Interest Rate
- Monthly Income
- Bankruptcy History
- Net Worth
- Loan Amount
- Credit Score

These coefficients represent model associations rather than causal relationships.

---

## 👥 Segment Analysis

Model behavior was examined across:

- Employment Status
- Education Level

False-positive and false-negative rates were also compared across Employment Status groups.

The analysis identified differences in error rates between groups, which should be investigated further through more comprehensive fairness testing before deployment.

---

## ⚠️ Limitations

Several limitations should be considered:

- Historical data may contain patterns that do not represent future lending conditions.
- The target variable is imbalanced.
- Some features contain missing values.
- Several numerical variables are strongly correlated.
- Model coefficients represent associations rather than causation.
- Segment analysis alone is not a complete fairness assessment.
- The model has not undergone prospective production validation.
- Regulatory, privacy, and compliance requirements would need additional assessment before deployment.

---

## 💡 Recommendations

The model should initially be used as a **decision-support tool** alongside human review rather than as the sole decision-maker.

Recommended next steps include:

- Monitor model performance after deployment.
- Monitor false-positive and false-negative rates.
- Review classification thresholds according to business risk appetite.
- Conduct comprehensive fairness testing.
- Validate the model on new applications.
- Monitor data and model drift.
- Periodically retrain and revalidate the model.
- Maintain appropriate explainability and compliance controls.

---

## 🚀 Future Improvements

Future work could include:

- Additional model algorithms and ensemble techniques
- More advanced feature engineering
- Alternative missing-value strategies
- Threshold optimization using business-cost scenarios
- More comprehensive fairness metrics
- Feature selection and dimensionality-reduction approaches
- Prospective validation using new loan applications
- Automated model monitoring and drift detection

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook
- Git
- GitHub

---

## 📁 Repository Contents

```text
financial-loan-risk-analysis/
│
├── financial_loan_risk.ipynb
├── financial_loan_data.csv
└── README.md