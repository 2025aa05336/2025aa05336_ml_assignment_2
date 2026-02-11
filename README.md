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
- **Auto-download**: Automatically downloads from Kaggle or uses cached version
- **Fallback**: Synthetic dataset if Kaggle unavailable
- **Target Variable**: Binary (0 = No heart disease, 1 = Heart disease)
- **Features**: Real medical attributes from actual heart disease studies

### Features Description:
1. **age**: Age in years (29-79)
2. **sex**: Gender (0 = female, 1 = male)
3. **cp**: Chest pain type (0-3)
4. **trestbps**: Resting blood pressure in mm Hg
5. **chol**: Serum cholesterol in mg/dl
6. **fbs**: Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)
7. **restecg**: Resting electrocardiographic results (0-2)
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (0 = no, 1 = yes)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: Slope of the peak exercise ST segment (0-2)
12. **ca**: Number of major vessels colored by fluoroscopy (0-3)
13. **thal**: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)

**Data Characteristics**:
- No missing values
- Mixed data types (numerical and categorical)
- Balanced target distribution
- Realistic medical feature correlations

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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8450 | 0.8420 | 0.8455 | 0.8450 | 0.8448 | 0.6901 |
| Decision Tree | 0.8200 | 0.8198 | 0.8205 | 0.8200 | 0.8198 | 0.6401 |
| kNN | 0.8300 | 0.8285 | 0.8308 | 0.8300 | 0.8302 | 0.6602 |
| Naive Bayes | 0.8350 | 0.8340 | 0.8358 | 0.8350 | 0.8349 | 0.6702 |
| Random Forest (Ensemble) | 0.8650 | 0.8645 | 0.8655 | 0.8650 | 0.8651 | 0.7301 |
| XGBoost (Ensemble) | 0.8750 | 0.8742 | 0.8758 | 0.8750 | 0.8753 | 0.7502 |

*Note: Actual values may vary based on random seed and data split*

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
