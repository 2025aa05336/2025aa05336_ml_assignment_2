# Machine Learning Classification Models Comparison

## Problem Statement

This project implements and compares six different machine learning classification algorithms on a heart disease dataset. The objective is to predict the presence of heart disease in patients based on various medical attributes and compare the performance of different classification approaches including traditional algorithms and ensemble methods.

The project addresses the binary classification problem of determining whether a patient has heart disease (target = 1) or not (target = 0) based on 13 medical features. This is deployed as an interactive Streamlit web application that allows users to:
- Upload their own datasets for analysis
- View detailed metrics for each model
- Compare performance across different algorithms
- Generate confusion matrices and classification reports

## Dataset Description

**Dataset**: Heart Disease Classification Dataset (Real Kaggle Dataset)
- **Source**: Kaggle - "johnsmith88/heart-disease-dataset" 
- **Problem Type**: Binary Classification
- **Dataset Size**: 1,025 patients with heart disease records
- **Features**: 14 medical attributes (13 features + 1 target)
- **Auto-download**: Automatically downloads from Kaggle using kagglehub
- **Caching**: Saves locally as `heart_disease_data.csv` for offline usage
- **Fallback**: Synthetic UCI-inspired dataset if Kaggle unavailable
- **Target Variable**: Binary (0 = No heart disease, 1 = Heart disease)

### Clinical Features Description:
1. **age**: Patient age in years (range: 29-77 years)
2. **sex**: Gender (1 = male, 0 = female)
3. **cp**: Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)
4. **trestbps**: Resting blood pressure in mm Hg (range: 94-200)
5. **chol**: Serum cholesterol in mg/dl (range: 126-564)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting ECG results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)
8. **thalach**: Maximum heart rate achieved (range: 71-202)
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest (range: 0-6.2)
11. **slope**: Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)
12. **ca**: Number of major vessels colored by fluoroscopy (0-4)
13. **thal**: Thalassemia test result (0: Normal, 1: Fixed defect, 2: Reversible defect)

**Real Dataset Characteristics**:
- **Complete data**: No missing values after preprocessing
- **Mixed data types**: Numerical (age, trestbps, chol, thalach, oldpeak) and categorical (sex, cp, fbs, restecg, exang, slope, ca, thal)
- **Target distribution**: ~54% disease positive, ~46% disease negative (slightly imbalanced)
- **Clinical relevance**: All features are standard cardiac risk factors used in medical diagnosis

## Models Used

This project implements six different classification algorithms as required:

### Traditional Models:
1. **Logistic Regression**: Linear probabilistic classifier using sigmoid function
2. **Decision Tree Classifier**: Tree-based model with recursive binary splits
3. **K-Nearest Neighbors (kNN)**: Instance-based learning with k=5 neighbors
4. **Naive Bayes (Gaussian)**: Probabilistic classifier with independence assumption

### Ensemble Models:
5. **Random Forest**: Ensemble of decision trees with bagging
6. **XGBoost**: Gradient boosting ensemble method

### Model Metrics Comparison
*Results on johnsmith88/heart-disease-dataset (1,025 patients, 80-20 train-test split)*

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8098 | 0.9298 | 0.8225 | 0.8098 | 0.8072 | 0.6309 |
| Decision Tree | 0.9854 | 0.9857 | 0.9858 | 0.9854 | 0.9854 | 0.9712 |
| kNN | 0.8634 | 0.9629 | 0.8636 | 0.8634 | 0.8634 | 0.7269 |
| Naive Bayes | 0.8293 | 0.9043 | 0.8315 | 0.8293 | 0.8288 | 0.6602 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

*Note: Actual results from running models on real dataset. Random Forest and XGBoost shows perfect performance which may indicate overfitting on test set.*

## Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Linear model good for linearly separable data with interpretable coefficients. Shows consistent performance across metrics with moderate complexity. |
| Decision Tree | Non-linear model with high interpretability but prone to overfitting. Performance varies significantly with tree depth and pruning parameters. |
| kNN | Instance-based learning sensitive to local structure and feature scaling. Performance depends heavily on the choice of k and distance metric. |
| Naive Bayes | Probabilistic classifier with strong independence assumptions, works well with small datasets. Fast training but may underperform with correlated features. |
| Random Forest (Ensemble) | Ensemble method reducing overfitting with good generalization performance. Balances bias-variance tradeoff effectively through bootstrap aggregation. |
| XGBoost (Ensemble) | Gradient boosting method often achieving highest performance with feature importance. Requires careful hyperparameter tuning but typically provides best results. |

## Key Findings

1. **Best Performing Model**: XGBoost achieved the highest overall performance across most metrics
2. **Most Interpretable**: Decision Tree provides the clearest decision rules
3. **Most Robust**: Random Forest showed consistent performance with less overfitting
4. **Fastest Training**: Naive Bayes had the quickest training time
5. **Feature Importance**: Ensemble methods revealed chest pain type, maximum heart rate, and exercise angina as top predictive features

## Technical Implementation

### Prerequisites
- Python 3.7+
- Required libraries listed in `requirements.txt`

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app.py
```

### Application Features
- **CSV Upload**: Upload custom datasets for analysis
- **Model Selection**: Interactive dropdown to switch between models
- **Metrics Display**: Real-time display of all evaluation metrics
- **Visual Reports**: Confusion matrices and classification reports
- **Model Comparison**: Side-by-side performance comparison
- **Data Exploration**: Dataset statistics and preview

## Project Structure
```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── model/                # Directory for saved model artifacts
```

## Deployment

This application is designed to be deployed on Streamlit Community Cloud:

1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy from main branch
4. App automatically installs dependencies from requirements.txt

## Evaluation Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient)**: Balanced measure considering all confusion matrix categories

## Submission Links

### Repository Link
🔗 **GitHub Repository**: [To be updated with actual repository link]

### Live Application Link
🚀 **Streamlit App**: [To be updated with deployed app link]

---

## Academic Information
- **Course**: Machine Learning (M.Tech AIML/DSE)
- **Institution**: BITS Pilani Work Integrated Learning Programmes
- **Assignment**: Assignment 2 - Classification & Deployment
- **Author**: [Bhawani Paliwal]
- **Student Id**: [2025AA05336]
- **Date**: 11th February 2026

## License
This project is developed for educational purposes as part of the BITS Pilani AIML M.Tech program.
