### Semiconductor Yield Prediction
 
A full end-to-end **machine learning pipeline** for predicting pass/fail outcomes in semiconductor manufacturing using signal sensor data.
 
**Pipeline Steps:**
- Data loading, inspection and missing value analysis
- Dropped columns with >50% missing values; imputed remaining with median
- Explored target class imbalance (Pass = -1, Fail = 1)
- EDA: histograms, boxplots, violin plots, correlation heatmaps
- **PCA** visualization to identify feature patterns
- Preprocessing pipeline: StandardScaler + OneHotEncoder + SimpleImputer
- **SMOTE** applied to handle class imbalance
- Compared three classifiers: **Random Forest**, **SVM** and **Naive Bayes**
- Evaluated using Accuracy, Precision, Recall, F1-score and ROC-AUC (5-fold cross-validation)
- Threshold tuning to maximize F1-score
- Confusion matrix, ROC curve and Precision-Recall curve visualizations
- Feature importance analysis (top 15 features)
- Saved best model using joblib

**Colab link:**
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/126vRqc64Mj908pWvgof8xVFqaPwJvAIp?usp=sharing)
 
**Best Performing Model:** Random Forest - highest recall, precision and ROC-AUC
 
**Tools & Libraries:** scikit-learn, imbalanced-learn (SMOTE), pandas, numpy, matplotlib, seaborn, joblib

*Author*

*Rekha Dhorigol*
