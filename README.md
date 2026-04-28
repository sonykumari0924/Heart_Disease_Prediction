# ❤️ AI Heart Disease Predictor

An end-to-end Machine Learning pipeline and interactive web application designed to predict the risk of heart disease based on clinical patient data. 

This project encompasses data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment via Streamlit, providing real-time inference for medical risk assessment.

## 🎯 Objective
To build a highly accurate, deployable supervised learning classifier that analyzes clinical parameters to identify patients at high risk of heart disease, streamlining preliminary medical decision-making.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (KNN, Logistic Regression, SVM, Naive Bayes, Decision Tree)
* **Data Visualization:** Matplotlib, Seaborn
* **Web Deployment:** Streamlit
* **Serialization:** Joblib

## ⚙️ System Architecture & Workflow

### 1. Data Processing & EDA
* Handled hidden missing values (e.g., zero-values in `Cholesterol` and `RestingBP`) using targeted mean imputation.
* Conducted extensive exploratory data analysis using heatmaps and distribution plots to identify feature correlations.
* Applied One-Hot Encoding (`pd.get_dummies`) to transform categorical medical variables (Chest Pain Type, Resting ECG, etc.) into machine-readable formats.
* Scaled numerical features using `StandardScaler` to optimize distance-based algorithms.

### 2. Model Evaluation
Multiple classifiers were trained and evaluated to find the optimal balance of Accuracy and F1-Score (prioritizing recall for medical diagnoses).
* **K-Nearest Neighbors (KNN):** Achieved highest performance (Accuracy: 88.59%, F1-Score: 89.86%).
* **Logistic Regression:** Accuracy: 87.50%
* **Naive Bayes:** Accuracy: 86.96%
* **SVM (RBF Kernel):** Accuracy: 86.41%

### 3. Streamlit Deployment
The highest-performing KNN model was serialized and deployed via Streamlit. The application dynamically handles user input, structures it into the exact expected feature schema, scales the data, and returns both a binary prediction and a probability confidence score.
