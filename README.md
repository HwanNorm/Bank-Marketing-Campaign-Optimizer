# Bank-Marketing-Campaign-Optimizer

A machine learning project to predict whether bank customers will subscribe to a term deposit using classification techniques.

## Project Overview

This project analyzes bank marketing data to build a predictive model for customer term deposit subscriptions. The analysis includes exploratory data analysis (EDA), basic classification modeling, and business impact evaluation.

## Dataset

- **Source**: Bank marketing dataset (CSV format with semicolon delimiter)
- **Size**: Not specified in code
- **Features**: 16 variables including demographics, financial information, and contact history
- **Target**: Binary classification (subscribe to term deposit: yes/no)

### Features Include:
- **Demographics**: age, job, marital status, education
- **Financial**: balance, housing loan, personal loan, default status
- **Contact**: contact type, day of week, month, duration, campaign details
- **Previous Campaign**: pdays, previous contacts, outcome

## Data Characteristics

- **Target Distribution**: Highly imbalanced (~89% no subscription, ~11% subscription)
- **Data Quality**: No missing values, no duplicate rows, no constant columns
- **Data Types**: Mix of numerical and categorical variables

## Analysis Performed

### Exploratory Data Analysis
- Target variable distribution analysis
- Numerical variables: descriptive statistics and correlation analysis
- Categorical variables: unique value counts and sample inspection
- Relationship analysis between features and target variable

### Key Findings from EDA
- Age shows some correlation with subscription likelihood (older customers more likely to subscribe)
- Most numerical features have weak correlation with target variable
- Duration variable has strong correlation but excluded due to data leakage concerns

### Feature Selection
- **Used**: Only numerical features for modeling
- **Excluded**: Categorical features (not processed), duration variable (data leakage)
- **Final Features**: age, balance, campaign, pdays, previous

## Modeling Approach

### Algorithm
- **Model**: Logistic Regression (basic implementation)
- **Library**: scikit-learn
- **Features**: Numerical variables only

### Data Splitting
- **Split**: 70% training, 30% testing
- **Method**: Random split (not stratified or time-based)

### Model Training
- Standard LogisticRegression() with default parameters
- No hyperparameter tuning performed
- No feature scaling or preprocessing applied

## Evaluation Results

### Classification Metrics
- Accuracy, Precision, Recall, F1-score calculated
- Classification report generated
- Confusion matrix analysis

### Model Performance Analysis
- **ROC-AUC**: Curve plotted to assess discriminative ability
- **Threshold Analysis**: Default 0.5 threshold used for predictions
- **Business Metrics**: Vintage table and lift analysis implemented

### Business Impact Analysis
- **Vintage Table**: Customers segmented into 10 probability bins
- **Lift Analysis**: Top-performing bin shows 3.8x higher subscription rate vs. overall average
- **Practical Application**: Analysis suggests targeting high-probability customers could improve campaign efficiency

## Limitations

1. **Feature Engineering**: No preprocessing of categorical variables
2. **Model Complexity**: Only basic logistic regression tested
3. **Class Imbalance**: No specific handling of imbalanced dataset
4. **Validation**: Simple train/test split, no cross-validation
5. **Feature Selection**: Limited to numerical features only

## Files

- `credit_score.py`: Main analysis script containing all code
- `bank.csv`: Dataset (semicolon-delimited)

## Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
```

## Usage

1. Ensure dataset `bank.csv` is in the same directory
2. Run `credit_score.py` to execute the full analysis pipeline
3. Review generated plots and metrics for model performance
4. Examine vintage table for business insights

## Results Summary

The project demonstrates a basic machine learning workflow for customer prediction. While the model shows some discriminative ability, the analysis reveals opportunities for improvement through feature engineering, advanced modeling techniques, and proper handling of class imbalance.

The business analysis suggests that model-based customer targeting could provide significant efficiency gains over random customer selection, with the top decile showing nearly 4x higher success rates.
