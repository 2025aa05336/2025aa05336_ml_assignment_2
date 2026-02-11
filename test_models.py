#!/usr/bin/env python3
"""
Test script for ML models using REAL heart disease dataset
This can be run to test the core functionality without Streamlit
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
# Handle XGBoost import gracefully
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None
    print(f"XGBoost not available (OpenMP/Import issue): {e}")

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                           recall_score, f1_score, matthews_corrcoef, 
                           confusion_matrix, classification_report)
import warnings

warnings.filterwarnings('ignore')

def load_heart_disease_dataset():
    """Load the real heart disease dataset"""
    # Check if cached locally
    local_dataset_path = "./heart_disease_data.csv"
    
    if os.path.exists(local_dataset_path):
        print("📁 Using cached Heart Disease dataset...")
        data = pd.read_csv(local_dataset_path)
    else:
        try:
            import kagglehub
            print("📥 Downloading Heart Disease dataset from Kaggle...")
            
            # Download latest version
            path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
            print(f"✅ Dataset downloaded to: {path}")
            
            # Find the CSV file in the downloaded path
            import glob
            csv_files = glob.glob(os.path.join(path, "*.csv"))
            
            if csv_files:
                data = pd.read_csv(csv_files[0])
                # Cache the dataset locally
                data.to_csv(local_dataset_path, index=False)
                print("💾 Dataset cached locally for future use.")
            else:
                print("❌ No CSV files found in downloaded dataset!")
                return create_fallback_dataset()
                
        except Exception as download_error:
            print(f"❌ Could not download from Kaggle: {download_error}")
            print("🔄 Using fallback synthetic dataset...")
            return create_fallback_dataset()
    
    # Standardize column names for the real dataset
    column_mapping = {
        'HeartDisease': 'target',
        'heart_disease': 'target', 
        'target': 'target',
        'output': 'target',
        'result': 'target',
        'class': 'target'
    }
    
    # Apply column mapping
    for old_name, new_name in column_mapping.items():
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})
            break
    
    # Ensure target column exists
    if 'target' not in data.columns:
        # Try to find the target column
        possible_targets = ['HeartDisease', 'heart_disease', 'output', 'result', 'class']
        for col in possible_targets:
            if col in data.columns:
                data = data.rename(columns={col: 'target'})
                break
        else:
            # If still no target, use the last column
            data = data.rename(columns={data.columns[-1]: 'target'})
    
    print(f"🎯 Real Heart Disease dataset loaded successfully!")
    print(f"📊 Dataset shape: {data.shape[0]} samples × {data.shape[1]} features")
    print(f"📈 Target distribution: {data['target'].value_counts().to_dict()}")
    
    return data

def create_fallback_dataset():
    """Generate synthetic dataset if real data unavailable"""
    print("🔄 Creating synthetic Heart Disease dataset...")
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    age = np.random.randint(29, 80, n_samples)
    sex = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.16, 0.29, 0.08])
    trestbps = np.random.normal(130, 17, n_samples).astype(int)
    chol = np.random.normal(246, 51, n_samples).astype(int)
    fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.52, 0.474, 0.006])
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

def preprocess_data(data):
    """Preprocess the dataset"""
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    print(f"🔧 Preprocessing data...")
    print(f"   • Original shape: {X.shape}")
    
    # Handle missing values
    if data.isnull().sum().sum() > 0:
        print(f"   • Found {data.isnull().sum().sum()} missing values, filling...")
        # Fill numerical columns with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].mode()[0])
    
    # Handle target variable encoding
    if y.dtype == 'object' or any(isinstance(val, str) for val in y.unique()):
        print(f"   • Converting target labels: {list(y.unique())}")
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        label_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
        print(f"   • Label mapping: {label_mapping}")
    
    # Handle categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_features:
        print(f"   • Encoding categorical features: {categorical_features}")
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    # Convert boolean features
    boolean_features = X.select_dtypes(include=['bool']).columns.tolist()
    if boolean_features:
        print(f"   • Converting boolean features: {boolean_features}")
        for col in boolean_features:
            X[col] = X[col].astype(int)
    
    print(f"✅ Preprocessing complete: {X.shape}")
    return X, y

def test_models():
    """Test all 6 models using REAL heart disease dataset"""
    print("🏥 REAL HEART DISEASE DATASET - ML MODEL EVALUATION")
    print("=" * 60)
    
    # Load real data
    data = load_heart_disease_dataset()
    if data is None:
        print("❌ Could not load dataset!")
        return None
    
    # Preprocess data
    X, y = preprocess_data(data)
    
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
    }
    
    # Add XGBoost or alternative
    if XGBOOST_AVAILABLE and XGBClassifier is not None:
        models['XGBoost (Ensemble)'] = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    else:
        print("⚠️ XGBoost not available, using Gradient Boosting alternative")
        from sklearn.ensemble import GradientBoostingClassifier
        models['Gradient Boosting (Alternative)'] = GradientBoostingClassifier(random_state=42, n_estimators=100)
    
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
    
    print("\n🎉 All models tested successfully locally!")
    
    return results

if __name__ == "__main__":
    test_models()
