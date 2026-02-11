import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
# Handle XGBoost import gracefully for local testing
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None
    print(f"XGBoost not available: {e}")

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                           recall_score, f1_score, matthews_corrcoef, 
                           confusion_matrix, classification_report)
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Models Comparison",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 Machine Learning Classification Models Comparison</h1>', unsafe_allow_html=True)

# Load default dataset from Kaggle
@st.cache
def load_default_dataset():
    """Load the real Heart Disease Classification dataset from Kaggle"""
    try:
        import kagglehub
        
        # Check if dataset already exists locally
        local_dataset_path = "./heart_disease_data.csv"
        
        if os.path.exists(local_dataset_path):
            st.info("📁 Using cached Heart Disease dataset...")
            data = pd.read_csv(local_dataset_path)
        else:
            # Download from Kaggle
            with st.spinner("📥 Downloading Heart Disease dataset from Kaggle... This may take a moment."):
                try:
                    # Download latest version
                    path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
                    st.success(f"✅ Dataset downloaded to: {path}")
                    
                    # Find the CSV file in the downloaded path
                    import glob
                    csv_files = glob.glob(os.path.join(path, "*.csv"))
                    
                    if csv_files:
                        data = pd.read_csv(csv_files[0])
                        # Cache the dataset locally
                        data.to_csv(local_dataset_path, index=False)
                        st.info("💾 Dataset cached locally for future use.")
                    else:
                        st.error("No CSV files found in downloaded dataset!")
                        return create_fallback_dataset()
                        
                except Exception as download_error:
                    st.warning(f"⚠️ Could not download from Kaggle: {download_error}")
                    st.info("Using fallback synthetic dataset...")
                    return create_fallback_dataset()
        
        # Standardize column names for the real dataset
        # Common variations in heart disease datasets
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
        
        st.success(f"🎯 Real Heart Disease dataset loaded successfully!")
        st.info(f"📊 Dataset shape: {data.shape[0]} samples × {data.shape[1]} features")
        
        return data
        
    except ImportError:
        st.warning("⚠️ kagglehub not available. Using fallback synthetic dataset.")
        return create_fallback_dataset()
    except Exception as e:
        st.warning(f"⚠️ Error loading Kaggle dataset: {e}")
        st.info("Using fallback synthetic dataset...")
        return create_fallback_dataset()

def create_fallback_dataset():
    """Create synthetic dataset if Kaggle download fails"""
    st.info("🔄 Creating synthetic Heart Disease dataset...")
    
    # Creating a synthetic heart disease dataset based on UCI repository
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

# Model training and evaluation functions
@st.cache(allow_output_mutation=True)
def train_models(X_train, X_test, y_train, y_test):
    """Train all available models and return results"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    # Add XGBoost only if available
    if XGBOOST_AVAILABLE and XGBClassifier is not None:
        models['XGBoost (Ensemble)'] = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    else:
        st.warning("⚠️ XGBoost not available locally (OpenMP issue). It will work on Streamlit Cloud deployment.")
        # Add an alternative ensemble model for local testing
        from sklearn.ensemble import GradientBoostingClassifier
        models['Gradient Boosting (Alternative)'] = GradientBoostingClassifier(random_state=42, n_estimators=100)
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle AUC score for multiclass
        if len(np.unique(y_test)) > 2:
            auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        else:
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.5
            
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'y_pred': y_pred,
            'y_test': y_test,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return results

def display_metrics(results):
    """Display metrics in a formatted table"""
    metrics_data = []
    for model_name, metrics in results.items():
        metrics_data.append({
            'ML Model Name': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'AUC': f"{metrics['auc']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1': f"{metrics['f1']:.4f}",
            'MCC': f"{metrics['mcc']:.4f}"
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics)
    
    return df_metrics

def create_confusion_matrix_plot(cm, model_name):
    """Create confusion matrix plot"""
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No Disease', 'Disease'],
                    y=['No Disease', 'Disease'],
                    color_continuous_scale='Blues',
                    title=f'Confusion Matrix - {model_name}')
    
    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
            )
    
    return fig

def create_metrics_comparison_plot(results):
    """Create a comparison plot of all metrics"""
    metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc']
    models = list(results.keys())
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[m.capitalize() for m in metrics],
        specs=[[{"secondary_y": False}]*3]*2
    )
    
    for i, metric in enumerate(metrics):
        row = i // 3 + 1
        col = i % 3 + 1
        
        values = [results[model][metric] for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric.capitalize(),
                   text=[f"{v:.3f}" for v in values],
                   textposition='auto'),
            row=row, col=col
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Model Metrics Comparison")
    return fig

# Main App
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Dataset Upload & Overview", "Model Training & Results", "Model Comparison"])
    
    if page == "Dataset Upload & Overview":
        st.markdown('<h2 class="sub-header">📊 Dataset Upload & Overview</h2>', unsafe_allow_html=True)
        
        # File upload option
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                data = load_default_dataset()
                st.info("Loading default Heart Disease dataset instead.")
        else:
            data = load_default_dataset()
            st.info("Using default Heart Disease dataset. Upload your own CSV to analyze different data.")
        
        # Dataset information
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Rows", data.shape[0])
        with col2:
            st.metric("Number of Features", data.shape[1])
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        # Display first few rows
        st.subheader("Dataset Preview")
        st.dataframe(data.head(10))
        
        # Dataset description
        st.subheader("Statistical Summary")
        st.dataframe(data.describe())
        
        # Store data in session state
        st.session_state.data = data
        
    elif page == "Model Training & Results":
        st.markdown('<h2 class="sub-header">🤖 Model Training & Results</h2>', unsafe_allow_html=True)
        
        if 'data' not in st.session_state:
            st.warning("Please upload a dataset first!")
            st.session_state.data = load_default_dataset()
        
        data = st.session_state.data
        
        # Smart target column detection
        target_col = None
        possible_targets = ['target', 'class', 'label', 'y', 'outcome', 'result']
        
        for col in possible_targets:
            if col in data.columns:
                target_col = col
                break
        
        # If no standard target column found, let user select
        if target_col is None:
            st.warning("⚠️ No standard target column found ('target', 'class', 'label', etc.)")
            target_col = st.selectbox("Select target column for classification:", data.columns.tolist())
            if st.button("Confirm Target Column"):
                st.session_state.target_confirmed = True
            
            if 'target_confirmed' not in st.session_state:
                st.info("Please select and confirm the target column to proceed.")
                return
        
        # Data preprocessing with comprehensive type handling
        st.subheader("🔧 Data Preprocessing")
        
        # Separate features and target
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        # Display data types
        with st.expander("📊 Data Type Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Feature Data Types:**")
                # Convert dtypes to string to avoid Arrow conversion issues
                dtypes_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Data Type': [str(dtype) for dtype in X.dtypes]
                })
                st.dataframe(dtypes_df)
            with col2:
                st.write("**Target Column Info:**")
                st.write(f"Column: {target_col}")
                st.write(f"Data Type: {str(y.dtype)}")
                st.write(f"Unique Values: {list(y.unique())}")
                # Convert value counts to avoid Arrow issues
                value_counts_df = pd.DataFrame({
                    'Value': y.value_counts().index.astype(str),
                    'Count': y.value_counts().values
                })
                st.write("**Value Counts:**")
                st.dataframe(value_counts_df)
        
        # Handle missing values
        if data.isnull().sum().sum() > 0:
            st.warning(f"⚠️ Found {data.isnull().sum().sum()} missing values. Filling with appropriate defaults...")
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
            
            # Handle target missing values
            if y.isnull().sum() > 0:
                st.error("❌ Target column has missing values. Please clean your data first.")
                return
        
        # Handle target variable encoding
        original_target_values = list(y.unique())
        n_classes = len(original_target_values)
        
        if y.dtype == 'object' or any(isinstance(val, str) for val in original_target_values):
            st.info(f"🔄 Converting string target labels to numerical")
            st.info(f"Original labels: {original_target_values}")
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
            
            # Create and display mapping
            label_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
            st.success(f"✅ Label mapping: {label_mapping}")
            y = pd.Series(y_encoded, index=y.index)
        else:
            # Check if numerical target needs encoding
            unique_vals = sorted(y.unique())
            if unique_vals != list(range(len(unique_vals))) or min(unique_vals) != 0:
                st.info(f"🔄 Normalizing numerical target labels: {unique_vals} → {list(range(len(unique_vals)))}")
                mapping_dict = {val: i for i, val in enumerate(unique_vals)}
                y = y.map(mapping_dict)
                st.success(f"✅ Target mapping: {mapping_dict}")
        
        # Classification type detection
        if n_classes == 2:
            st.info(f"🎯 Detected **Binary Classification** problem ({n_classes} classes)")
        else:
            st.info(f"🎯 Detected **Multi-class Classification** problem ({n_classes} classes)")
        
        # Handle feature encoding
        preprocessing_info = []
        
        # Identify data types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        boolean_features = X.select_dtypes(include=['bool']).columns.tolist()
        
        # Convert boolean to numeric
        if boolean_features:
            preprocessing_info.append(f"Converting boolean features to numeric: {boolean_features}")
            for col in boolean_features:
                X[col] = X[col].astype(int)
            numeric_features.extend(boolean_features)
        
        # Encode categorical features
        if categorical_features:
            preprocessing_info.append(f"Encoding categorical features: {categorical_features}")
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                # Convert to numeric type
                X[col] = pd.to_numeric(X[col])
            numeric_features.extend(categorical_features)
        
        # Display preprocessing summary
        if preprocessing_info:
            st.info("🏷️ **Feature Preprocessing:**")
            for info in preprocessing_info:
                st.write(f"  • {info}")
        
        # Final data validation
        st.success(f"✅ **Data Ready for Training:**")
        st.write(f"  • Features: {X.shape[1]} columns, {X.shape[0]} rows")
        st.write(f"  • Target: {n_classes} classes, {len(y)} samples")
        st.write(f"  • All features are now numeric: {X.dtypes.apply(lambda x: 'numeric' if np.issubdtype(x, np.number) else 'other').unique()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        with st.spinner("Training models... This may take a few moments."):
            results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        st.success("All models trained successfully!")
        
        # Model selection dropdown
        st.subheader("Model Selection")
        selected_model = st.selectbox("Choose a model to view details:", list(results.keys()))
        
        # Display metrics for selected model
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Metrics for {selected_model}")
            metrics = results[selected_model]
            
            st.markdown(f"""
            <div class="metric-card">
                <b>Accuracy:</b> {metrics['accuracy']:.4f}<br>
                <b>AUC Score:</b> {metrics['auc']:.4f}<br>
                <b>Precision:</b> {metrics['precision']:.4f}<br>
                <b>Recall:</b> {metrics['recall']:.4f}<br>
                <b>F1 Score:</b> {metrics['f1']:.4f}<br>
                <b>MCC Score:</b> {metrics['mcc']:.4f}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm_fig = create_confusion_matrix_plot(metrics['confusion_matrix'], selected_model)
            st.plotly_chart(cm_fig)
        
        # Classification Report
        st.subheader("Classification Report")
        y_test = results[selected_model]['y_test']
        y_pred = results[selected_model]['y_pred']
        report = classification_report(y_test, y_pred, output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Store results in session state
        st.session_state.results = results
        
    elif page == "Model Comparison":
        st.markdown('<h2 class="sub-header">📈 Model Comparison</h2>', unsafe_allow_html=True)
        
        if 'results' not in st.session_state:
            st.warning("Please train the models first!")
            return
        
        results = st.session_state.results
        
        # Metrics comparison table
        st.subheader("Model Metrics Comparison Table")
        metrics_df = display_metrics(results)
        
        # Download metrics table
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="Download Metrics as CSV",
            data=csv,
            file_name="model_metrics_comparison.csv",
            mime="text/csv"
        )
        
        # Visual comparison
        st.subheader("Visual Metrics Comparison")
        comparison_fig = create_metrics_comparison_plot(results)
        st.plotly_chart(comparison_fig)
        
        # Performance observations
        st.subheader("Performance Observations")
        
        observations = {
            'Logistic Regression': 'Linear model good for linearly separable data with interpretable coefficients.',
            'Decision Tree': 'Non-linear model with high interpretability but prone to overfitting.',
            'kNN': 'Instance-based learning sensitive to local structure and feature scaling.',
            'Naive Bayes': 'Probabilistic classifier with strong independence assumptions, works well with small datasets.',
            'Random Forest (Ensemble)': 'Ensemble method reducing overfitting with good generalization performance.',
            'XGBoost (Ensemble)': 'Gradient boosting method often achieving highest performance with feature importance.'
        }
        
        obs_data = []
        for model_name, obs in observations.items():
            accuracy = results[model_name]['accuracy']
            obs_data.append({
                'ML Model Name': model_name,
                'Accuracy': f"{accuracy:.4f}",
                'Observation about model performance': obs
            })
        
        obs_df = pd.DataFrame(obs_data)
        st.dataframe(obs_df)

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎓 Machine Learning Assignment 2 - Classification & Deployment</p>
        <p>Built with Streamlit • Developed for BITS Pilani</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()
