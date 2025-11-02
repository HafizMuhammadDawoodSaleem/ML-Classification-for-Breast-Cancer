Breast Cancer Classification Using Machine Learning

Prepared By: Dawood Saleem

E-Mail:  dawoodsaleem410@gmail.com

Abstract

This project focuses on classifying breast cancer tumors into malignant or benign categories using supervised machine learning algorithms.
The study utilizes the Wisconsin Breast Cancer Dataset to train and evaluate multiple models, comparing their performance using key evaluation metrics.
The main goal is to design a model that assists healthcare professionals in early cancer detection through data-driven insights.

Introduction

Breast cancer remains one of the most life-threatening diseases among women worldwide.
Early detection significantly improves the chances of survival.
By leveraging data science and machine learning, this project applies classification algorithms to analyze tumor features and predict whether a tumor is malignant or benign.

The integration of predictive modeling helps enhance diagnostic accuracy and supports timely medical interventions.

Dataset Description

The dataset used in this project is the Wisconsin Breast Cancer Dataset (from Kaggle), containing 569 samples with 30 numerical features describing cell nucleus properties.
Each record is labeled as either:

Malignant (1) — Cancerous

Benign (0) — Non-cancerous

Key features include:

Mean radius

Mean texture

Mean smoothness

Compactness

Symmetry

All features were derived from digital images of breast mass cells.

Libraries Used

NumPy — Numerical computation

Pandas — Data manipulation and cleaning

Matplotlib & Seaborn — Visualization and correlation analysis

Scikit-learn — Model implementation (Logistic Regression, Random Forest, etc.)

Methodology

Data Preprocessing

Cleaned dataset and handled missing values.

Standardized numerical features for better model performance.

Exploratory Data Analysis (EDA)

Visualized relationships between features using heatmaps, histograms, and pair plots.

Model Training

Trained models such as Logistic Regression and Random Forest.

Evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

Evaluation and Optimization

Compared multiple classifiers to identify the most robust model.

Applied hyperparameter tuning (e.g., GridSearchCV) for performance improvement.

Results and Discussion

The Random Forest Classifier achieved the highest accuracy (~98%), outperforming other models.
It maintained a strong balance between sensitivity and specificity, with minimal false positives and false negatives.
Visualization of the confusion matrix and ROC curve confirmed the reliability and stability of the model.

Conclusion

This project demonstrates the potential of machine learning in medical diagnosis, particularly in breast cancer detection.
By using the Random Forest model, the system achieved high accuracy and dependable classification between malignant and benign tumors.
Future enhancements may include:

Integration of deep learning models.

Expansion with more diverse clinical datasets.

Deployment as an interactive healthcare tool.

This project highlights how data-driven methods can empower medical decision-making and improve patient outcomes.
