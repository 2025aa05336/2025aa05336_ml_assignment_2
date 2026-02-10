#!/usr/bin/env python3
"""
Simple test script to verify that all ML models work correctly
This can be run to test the core functionality without Streamlit
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, matthews_corrcoef, 
                           confusion_matrix, classification_report)
import warnings

warnings.filterwarnings('ignore')

def generate_test_data():
    """Generate test dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    age = np.random.randint(29, 80, n_samples)
    sex = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.16, 0.29, 0.08])
    trestbps = np.random.normal(130, 17, n_samples).astype(int)
    chol = np.random.normal(246, 51, n_samples).astype(int)
    fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.52, 0.48, 0.006])
    thalach = np.random.normal(150, 22, n_samples).astype(int)
    exang = np.random.choice([0, 1], n_samples, p=[0.68, 0.32])
    oldpeak = np.random.exponential(1, n_samples).round(1)
    slope = np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.14, 0.65])
    ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.59, 0.23, 0.13, 0.05])
    thal = np.random.choice([1, 2, 3], n_samples, p=[0.02, 0.55, 0.43])
    
    # Generate target with realistic correlation
    target_prob = (
        0.1 * (age > 60) +
        0.2 * sex +
        0.15 * (cp >= 2) +
        0.1 * (trestbps > 140) +
        0.1 * (chol > 240) +
        0.05 * fbs +
        0.05 * restecg +
        0.15 * (thalach < 130) +
        0.2 * exang +
        0.15 * (oldpeak > 2) +
        0.1 * (slope == 0) +
        0.2 * (ca > 0) +
        0.15 * (thal == 3)
    )
    target = np.random.binomial(1, np.clip(target_prob, 0.1, 0.9), n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': target
    })
    
    return data

def test_models():
    """Test all 6 models and print results"""
    print("🧪 Testing Machine Learning Models")
    print("=" * 50)
    
    # Load data
    print("📊 Loading test data...")
    data = generate_test_data()
    print(f"✅ Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
    
    # Prepare features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📈 Training set: {X_train_scaled.shape[0]} samples")
    print(f"📊 Test set: {X_test_scaled.shape[0]} samples")
    print()
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost (Ensemble)': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    }
    
    print("🤖 Training and evaluating models...")
    print()
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.5
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc
        }
        
        print(f"✅ {name} completed")
    
    print("\n📊 RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model Name':<25} {'Accuracy':<10} {'AUC':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'MCC':<8}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['auc']:<8.4f} "
              f"{metrics['precision']:<10.4f} {metrics['recall']:<8.4f} "
              f"{metrics['f1']:<8.4f} {metrics['mcc']:<8.4f}")
    
    print("\n🎉 All models tested successfully!")
    print("\n📝 Next steps:")
    print("1. Run 'streamlit run app.py' to launch the web application")
    print("2. Upload your own dataset or use the default heart disease dataset")
    print("3. Compare model performances interactively")
    
    return results

if __name__ == "__main__":
    test_models()
