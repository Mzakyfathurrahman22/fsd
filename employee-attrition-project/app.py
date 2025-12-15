"""
==========================================
APLIKASI PREDIKSI EMPLOYEE ATTRITION
==========================================
Aplikasi Machine Learning untuk memprediksi
karyawan yang berisiko resign (attrition)

Dibuat dengan: Streamlit + Scikit-learn
==========================================
"""

# ============================================
# 1. IMPORT LIBRARIES
# ============================================

import streamlit as st  # Framework untuk buat web app
import pandas as pd  # Untuk manipulasi data (seperti Excel di Python)
import numpy as np  # Untuk operasi matematika dan array
import matplotlib.pyplot as plt  # Untuk visualisasi grafik
import seaborn as sns  # Untuk visualisasi yang lebih cantik
import plotly.express as px  # Untuk visualisasi interaktif
import plotly.graph_objects as go  # Untuk grafik plotly advanced
from plotly.subplots import make_subplots  # Untuk multiple subplot

# Library untuk Machine Learning
from sklearn.model_selection import train_test_split  # Untuk split data train/test
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Untuk preprocessing
from sklearn.linear_model import LogisticRegression  # Algoritma 1
from sklearn.tree import DecisionTreeClassifier  # Algoritma 2
from sklearn.ensemble import RandomForestClassifier  # Algoritma 3
from sklearn.svm import SVC  # Algoritma 4
from xgboost import XGBClassifier  # Algoritma 5

# Library untuk evaluasi model
from sklearn.metrics import (
    accuracy_score,  # Untuk hitung accuracy
    precision_score,  # Untuk hitung precision
    recall_score,  # Untuk hitung recall
    f1_score,  # Untuk hitung f1-score
    confusion_matrix,  # Untuk confusion matrix
    classification_report,  # Untuk report lengkap
    roc_curve,  # Untuk ROC curve
    roc_auc_score  # Untuk AUC score
)


import joblib  # Untuk save/load model
import warnings
warnings.filterwarnings('ignore')  # Sembunyikan warning agar tampilan bersih

# ============================================
# 2. KONFIGURASI STREAMLIT PAGE
# ============================================

st.set_page_config(
    page_title="Employee Attrition Predictor",  # Judul tab browser
    layout="wide",  # Pakai layout lebar (full width)
    initial_sidebar_state="expanded"  # Sidebar terbuka otomatis
)

# ============================================
# 3. CUSTOM CSS STYLING
# ============================================
# CSS untuk bikin tampilan lebih menarik

st.markdown("""
    <style>
    /* Style untuk main header */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Style untuk sub-header */
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Style untuk metric cards */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    /* Style untuk info box */
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Style untuk warning box */
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    /* Style untuk success box */
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 4. FUNGSI HELPER - LOAD DATA
# ============================================

@st.cache_data  # Cache data agar tidak perlu load ulang setiap kali refresh
def load_data():
    """
    Fungsi untuk load dataset dari file CSV
    
    Returns:
        df (DataFrame): Dataset yang sudah di-load
    """
    try:
        # Coba load dari file lokal
        df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
        return df
    except:
        # Jika file tidak ada, tampilkan pesan error
        st.error("⚠️ Dataset tidak ditemukan! Silakan download dataset dari Kaggle.")
        st.markdown("""
        **Download Dataset:**
        1. Kunjungi: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
        2. Download file CSV
        3. Simpan di folder `data/` dengan nama `WA_Fn-UseC_-HR-Employee-Attrition.csv`
        """)
        return None

# ============================================
# 5. FUNGSI HELPER - PREPROCESSING
# ============================================

def preprocess_data(df):
    """
    Fungsi untuk preprocessing data:
    1. Encode categorical variables (ubah text jadi angka)
    2. Handle missing values (jika ada data kosong)
    3. Feature scaling (normalisasi angka)
    
    Args:
        df (DataFrame): Dataset mentah
        
    Returns:
        df_processed (DataFrame): Dataset yang sudah diproses
        label_encoders (dict): Dictionary berisi encoder untuk setiap kolom
        scaler (StandardScaler): Scaler untuk normalisasi
    """
    
    # Copy dataframe agar tidak mengubah data original
    df_processed = df.copy()
    
    # Dictionary untuk simpan label encoder setiap kolom
    label_encoders = {}
    
    # 1. HANDLE MISSING VALUES
    # Cek apakah ada data kosong
    if df_processed.isnull().sum().sum() > 0:
        st.warning("⚠️ Ditemukan missing values, akan diisi dengan median/mode")
        # Isi numeric columns dengan median
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        # Isi categorical columns dengan mode (nilai paling sering muncul)
        cat_cols = df_processed.select_dtypes(include=['object']).columns
        df_processed[cat_cols] = df_processed[cat_cols].fillna(df_processed[cat_cols].mode().iloc[0])
    
    # 2. ENCODE CATEGORICAL VARIABLES
    # Cari semua kolom yang bertipe object (text/string)
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        # Buat encoder untuk setiap kolom
        le = LabelEncoder()
        # Fit dan transform kolom tersebut
        df_processed[col] = le.fit_transform(df_processed[col])
        # Simpan encoder untuk digunakan nanti saat prediksi
        label_encoders[col] = le
    
    # 3. FEATURE SCALING
    # Ambil semua kolom numeric kecuali target (Attrition)
    feature_columns = [col for col in df_processed.columns if col != 'Attrition']
    
    # Buat scaler
    scaler = StandardScaler()
    # Fit dan transform data
    df_processed[feature_columns] = scaler.fit_transform(df_processed[feature_columns])
    
    return df_processed, label_encoders, scaler

# ============================================
# 6. FUNGSI HELPER - TRAIN MODELS
# ============================================

def train_models(X_train, X_test, y_train, y_test):
    """
    Fungsi untuk training 5 model sekaligus
    
    Args:
        X_train: Features training data
        X_test: Features testing data
        y_train: Target training data
        y_test: Target testing data
        
    Returns:
        models (dict): Dictionary berisi semua model yang sudah di-train
        results (dict): Dictionary berisi hasil evaluasi semua model
    """
    
    # Dictionary untuk simpan model
    models = {}
    
    # Dictionary untuk simpan hasil evaluasi
    results = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'AUC': []
    }
    
    # Progress bar untuk tampilan training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # ========================================
    # MODEL 1: LOGISTIC REGRESSION
    # ========================================
    status_text.text("🔄 Training Logistic Regression...")
    
    # Inisialisasi model
    lr_model = LogisticRegression(
        max_iter=1000,  # Maksimal iterasi
        random_state=42,  # Seed untuk reproducibility
        class_weight='balanced'  # Handle imbalanced data
    )
    
    # Training model
    lr_model.fit(X_train, y_train)
    
    # Prediksi
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]  # Probabilitas untuk class 1
    
    # Hitung metrics
    results['Model'].append('Logistic Regression')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_lr))
    results['Precision'].append(precision_score(y_test, y_pred_lr))
    results['Recall'].append(recall_score(y_test, y_pred_lr))
    results['F1-Score'].append(f1_score(y_test, y_pred_lr))
    results['AUC'].append(roc_auc_score(y_test, y_pred_proba_lr))
    
    # Simpan model
    models['Logistic Regression'] = lr_model
    
    progress_bar.progress(20)
    
    # ========================================
    # MODEL 2: DECISION TREE
    # ========================================
    status_text.text("🔄 Training Decision Tree...")
    
    dt_model = DecisionTreeClassifier(
        max_depth=10,  # Kedalaman maksimal tree
        min_samples_split=20,  # Minimal sampel untuk split
        min_samples_leaf=10,  # Minimal sampel di leaf
        random_state=42,
        class_weight='balanced'
    )
    
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
    
    results['Model'].append('Decision Tree')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_dt))
    results['Precision'].append(precision_score(y_test, y_pred_dt))
    results['Recall'].append(recall_score(y_test, y_pred_dt))
    results['F1-Score'].append(f1_score(y_test, y_pred_dt))
    results['AUC'].append(roc_auc_score(y_test, y_pred_proba_dt))
    
    models['Decision Tree'] = dt_model
    
    progress_bar.progress(40)
    
    # ========================================
    # MODEL 3: RANDOM FOREST (BIASANYA TERBAIK!)
    # ========================================
    status_text.text("🔄 Training Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Jumlah trees
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  # Gunakan semua CPU cores
    )
    
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    results['Model'].append('Random Forest')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_rf))
    results['Precision'].append(precision_score(y_test, y_pred_rf))
    results['Recall'].append(recall_score(y_test, y_pred_rf))
    results['F1-Score'].append(f1_score(y_test, y_pred_rf))
    results['AUC'].append(roc_auc_score(y_test, y_pred_proba_rf))
    
    models['Random Forest'] = rf_model
    
    progress_bar.progress(60)
    
    # ========================================
    # MODEL 4: SUPPORT VECTOR MACHINE (SVM)
    # ========================================
    status_text.text("🔄 Training SVM...")
    
    svm_model = SVC(
        kernel='rbf',  # Radial Basis Function kernel
        C=1.0,  # Regularization parameter
        gamma='scale',
        probability=True,  # Enable probability prediction
        random_state=42,
        class_weight='balanced'
    )
    
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]
    
    results['Model'].append('SVM')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_svm))
    results['Precision'].append(precision_score(y_test, y_pred_svm))
    results['Recall'].append(recall_score(y_test, y_pred_svm))
    results['F1-Score'].append(f1_score(y_test, y_pred_svm))
    results['AUC'].append(roc_auc_score(y_test, y_pred_proba_svm))
    
    models['SVM'] = svm_model
    
    progress_bar.progress(80)
    
    # ========================================
    # MODEL 5: XGBOOST (ADVANCED!)
    # ========================================
    status_text.text("🔄 Training XGBoost...")
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Handle imbalanced data dengan scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model.set_params(scale_pos_weight=scale_pos_weight)
    
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    results['Model'].append('XGBoost')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_xgb))
    results['Precision'].append(precision_score(y_test, y_pred_xgb))
    results['Recall'].append(recall_score(y_test, y_pred_xgb))
    results['F1-Score'].append(f1_score(y_test, y_pred_xgb))
    results['AUC'].append(roc_auc_score(y_test, y_pred_proba_xgb))
    
    models['XGBoost'] = xgb_model
    
    progress_bar.progress(100)
    status_text.text("✅ Training selesai!")
    
    return models, results

# ============================================
# 7. FUNGSI HELPER - PLOT CONFUSION MATRIX
# ============================================

def plot_confusion_matrix(y_test, y_pred, model_name):
    """
    Fungsi untuk plot confusion matrix dengan visualisasi yang menarik
    
    Args:
        y_test: Target testing data (actual)
        y_pred: Prediksi model
        model_name: Nama model
    """
    
    # Hitung confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Buat figure dengan plotly
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Stay', 'Predicted: Resign'],
        y=['Actual: Stay', 'Actual: Resign'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan penjelasan
    st.markdown(f"""
    <div class="info-box">
    <b>📊 Penjelasan Confusion Matrix:</b><br>
    • <b>True Negative (TN) = {cm[0][0]}</b>: Prediksi STAY, Aktual STAY ✅<br>
    • <b>False Positive (FP) = {cm[0][1]}</b>: Prediksi RESIGN, Aktual STAY ❌<br>
    • <b>False Negative (FN) = {cm[1][0]}</b>: Prediksi STAY, Aktual RESIGN ❌<br>
    • <b>True Positive (TP) = {cm[1][1]}</b>: Prediksi RESIGN, Aktual RESIGN ✅
    </div>
    """, unsafe_allow_html=True)

# ============================================
# 8. FUNGSI HELPER - PLOT ROC CURVE
# ============================================

def plot_roc_curves(models, X_test, y_test):
    """
    Fungsi untuk plot ROC curves untuk semua model
    
    Args:
        models: Dictionary berisi semua model
        X_test: Features testing data
        y_test: Target testing data
    """
    
    fig = go.Figure()
    
    # Plot ROC curve untuk setiap model
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{name} (AUC = {auc:.3f})',
            mode='lines'
        ))
    
    # Plot diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves - All Models',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
    <b>📈 Penjelasan ROC Curve:</b><br>
    • <b>ROC Curve</b> menunjukkan trade-off antara True Positive Rate (Recall) dan False Positive Rate<br>
    • <b>AUC (Area Under Curve)</b> mengukur seberapa baik model membedakan antara 2 class<br>
    • <b>AUC = 1.0</b>: Model sempurna<br>
    • <b>AUC = 0.5</b>: Model seperti tebak random<br>
    • Semakin tinggi AUC, semakin bagus model
    </div>
    """, unsafe_allow_html=True)

# ============================================
# 9. FUNGSI HELPER - FEATURE IMPORTANCE
# ============================================

def plot_feature_importance(model, feature_names, model_name):
    """
    Fungsi untuk plot feature importance
    (hanya untuk tree-based models)
    
    Args:
        model: Trained model
        feature_names: Nama-nama features
        model_name: Nama model
    """
    
    # Cek apakah model punya feature importance
    if hasattr(model, 'feature_importances_'):
        # Ambil feature importance
        importances = model.feature_importances_
        
        # Buat DataFrame untuk sorting
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)  # Ambil top 15
        
        # Plot dengan plotly
        fig = px.bar(
            feature_imp_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top 15 Feature Importance - {model_name}',
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <b>🔍 Penjelasan Feature Importance:</b><br>
        • Menunjukkan seberapa penting setiap feature dalam membuat prediksi<br>
        • Semakin tinggi nilai, semakin besar pengaruhnya terhadap keputusan resign<br>
        • Berguna untuk memahami faktor apa yang paling mempengaruhi attrition
        </div>
        """, unsafe_allow_html=True)

# ============================================
# 10. SIDEBAR NAVIGATION
# ============================================

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

# Menu navigasi
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Home", "📊 Data Exploration", "🤖 Model Training", "🎯 Prediction", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
        <h4>📚 Proyek Tugas</h4>
        <p><b>Employee Attrition Prediction</b></p>
        <p>Machine Learning Application</p>
    </div>
""", unsafe_allow_html=True)

# ============================================
# 11. PAGE: HOME
# ============================================

if page == "🏠 Home":
    # Header
    st.markdown('<h1 class="main-header">👔 Employee Attrition Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Prediksi Karyawan yang Berisiko Resign menggunakan Machine Learning</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3 kolom untuk overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Tujuan</h3>
            <p>Memprediksi karyawan yang berisiko resign untuk pencegahan dini</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Teknologi</h3>
            <p>5 algoritma ML: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dataset</h3>
            <p>IBM HR Analytics: 1,470 karyawan dengan 35 features</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Konteks Permasalahan
    st.header("📋 1. Konteks Permasalahan")
    
    st.markdown("""
    <div class="info-box">
    <h4>❓ Apa Masalahnya?</h4>
    <p>
    Perusahaan menghadapi masalah <b>employee attrition (karyawan resign)</b> yang tinggi. 
    Kehilangan karyawan berpengalaman dapat menyebabkan:
    </p>
    <ul>
        <li>💰 <b>Biaya recruitment tinggi</b>: Iklan lowongan, proses interview, onboarding</li>
        <li>⏰ <b>Waktu training</b>: Karyawan baru butuh waktu untuk produktif</li>
        <li>📉 <b>Penurunan produktivitas</b>: Knowledge loss dan gap dalam tim</li>
        <li>🔄 <b>Turnover effect</b>: Karyawan lain jadi ikut resign</li>
    </ul>
    <p>
    Menurut penelitian, <b>biaya kehilangan 1 karyawan = 1.5-2x gaji tahunan</b> mereka!
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <h4>💡 Solusi</h4>
    <p>
    Membangun sistem prediksi berbasis Machine Learning yang dapat:
    </p>
    <ul>
        <li>🎯 <b>Identifikasi dini</b> karyawan yang berisiko resign</li>
        <li>📊 <b>Analisis faktor</b> apa yang mempengaruhi keputusan resign</li>
        <li>⚡ <b>Intervensi cepat</b> dari HRD (raise gaji, promosi, work-life balance)</li>
        <li>💼 <b>Retention strategy</b> yang lebih efektif dan terukur</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Urgency</h4>
    <p>
    Sistem ini <b>urgent</b> karena:
    </p>
    <ul>
        <li>Attrition rate tinggi = kerugian finansial besar</li>
        <li>Kompetisi talent semakin ketat</li>
        <li>Dampak langsung ke operasional dan moral tim</li>
        <li>Prevention lebih murah daripada replacement</li>
    </ul>
    <p><b>Target User:</b> HRD Manager, People Operations, Department Head</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fitur-Fitur Aplikasi
    st.header("✨ 2. Fitur-Fitur Aplikasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Data Exploration
        - Visualisasi distribusi data
        - Analisis statistik deskriptif
        - Correlation heatmap
        - Insight dari data
        
        ### 🤖 Model Training
        - Training 5 algoritma ML
        - Model comparison
        - Hyperparameter tuning
        - Performance evaluation
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Prediction
        - Input data karyawan
        - Real-time prediction
        - Probability score
        - Risk assessment
        
        ### 📈 Analytics
        - Confusion Matrix
        - ROC Curve
        - Feature Importance
        - Detailed metrics
        """)
    
    st.markdown("---")
    
    # Metodologi
    st.header("🔬 3. Metodologi")
    
    st.markdown("""
    <div class="info-box">
    <h4>📝 Tahapan Pengembangan:</h4>
    <ol>
        <li><b>Data Collection</b>: Dataset IBM HR Analytics (Secondary Data)</li>
        <li><b>Data Preprocessing</b>:
            <ul>
                <li>Handle missing values</li>
                <li>Encode categorical variables (Label Encoding)</li>
                <li>Feature scaling (StandardScaler)</li>
            </ul>
        </li>
        <li><b>Train-Test Split</b>: 70% Training, 30% Testing</li>
        <li><b>Model Training</b>: 5 algoritma (LR, DT, RF, SVM, XGBoost)</li>
        <li><b>Model Evaluation</b>: Accuracy, Precision, Recall, F1-Score, AUC</li>
        <li><b>Model Selection</b>: Pilih model terbaik berdasarkan metrics</li>
        <li><b>Deployment</b>: Streamlit web application</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# 12. PAGE: DATA EXPLORATION
# ============================================

elif page == "📊 Data Exploration":
    st.markdown('<h1 class="main-header">📊 Data Exploration</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Overview
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            attrition_rate = (df['Attrition'] == 'Yes').sum() / len(df) * 100
            st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # Show dataset
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv,
            file_name="employee_data.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Data Info
        st.subheader("📊 Data Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tipe Data:**")
            st.dataframe(df.dtypes.astype(str).rename("Data Type").to_frame(), use_container_width=True)
        
        with col2:
            st.write("**Statistik Deskriptif:**")
            st.dataframe(df.describe().round(2), use_container_width=True)
        
        st.markdown("---")
        
        # Visualisasi
        st.header("📈 Data Visualization")
        
        # 1. Attrition Distribution
        st.subheader("1️⃣ Distribusi Attrition")
        
        attrition_counts = df['Attrition'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=attrition_counts.index,
                y=attrition_counts.values,
                text=attrition_counts.values,
                textposition='auto',
                marker_color=['#2ecc71', '#e74c3c']
            )
        ])
        
        fig.update_layout(
            title='Distribusi Karyawan: Stay vs Resign',
            xaxis_title='Status',
            yaxis_title='Jumlah Karyawan',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        📊 **Insight**: 
        - Karyawan yang STAY: {attrition_counts['No']} ({attrition_counts['No']/len(df)*100:.1f}%)
        - Karyawan yang RESIGN: {attrition_counts['Yes']} ({attrition_counts['Yes']/len(df)*100:.1f}%)
        - Dataset ini **imbalanced** (tidak seimbang), makanya kita perlu handle dengan teknik khusus
        """)
        
        st.markdown("---")
        
        # 2. Attrition by Department
        st.subheader("2️⃣ Attrition berdasarkan Department")
        
        dept_attrition = pd.crosstab(df['Department'], df['Attrition'], normalize='index') * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=dept_attrition.index,
            y=dept_attrition['No'],
            name='Stay',
            marker_color='#2ecc71'
        ))
        
        fig.add_trace(go.Bar(
            x=dept_attrition.index,
            y=dept_attrition['Yes'],
            name='Resign',
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title='Attrition Rate by Department',
            xaxis_title='Department',
            yaxis_title='Percentage (%)',
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 3. Attrition by Age Group
        st.subheader("3️⃣ Attrition berdasarkan Age Group")
        
        # Buat age groups
        df_viz = df.copy()
        df_viz['AgeGroup'] = pd.cut(df_viz['Age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
        
        age_attrition = pd.crosstab(df_viz['AgeGroup'], df_viz['Attrition'], normalize='index') * 100
        
        fig = px.bar(
            age_attrition,
            barmode='group',
            title='Attrition Rate by Age Group',
            labels={'value': 'Percentage (%)', 'AgeGroup': 'Age Group'},
            color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 4. Salary vs Attrition
        st.subheader("4️⃣ Monthly Income vs Attrition")
        
        fig = px.box(
            df,
            x='Attrition',
            y='MonthlyIncome',
            color='Attrition',
            title='Distribusi Gaji: Stay vs Resign',
            color_discrete_map={'No': '#2ecc71', 'Yes': '#e74c3c'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        avg_stay = df[df['Attrition'] == 'No']['MonthlyIncome'].mean()
        avg_resign = df[df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
        
        st.info(f"""
        📊 **Insight**: 
        - Rata-rata gaji yang STAY: ${avg_stay:,.0f}
        - Rata-rata gaji yang RESIGN: ${avg_resign:,.0f}
        - Perbedaan: ${avg_stay - avg_resign:,.0f}
        - Karyawan dengan gaji lebih rendah cenderung resign!
        """)
        
        st.markdown("---")
        
        # 5. Correlation Heatmap
        st.subheader("5️⃣ Correlation Heatmap")
        
        # Pilih numeric columns saja
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Correlation Matrix - Numeric Features'
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <b>📊 Cara Baca Correlation Heatmap:</b><br>
        • <b>1.0</b> = Korelasi positif sempurna (naik bareng)<br>
        • <b>0.0</b> = Tidak ada korelasi<br>
        • <b>-1.0</b> = Korelasi negatif sempurna (berlawanan)<br>
        • Warna <b>merah</b> = korelasi positif, <b>biru</b> = korelasi negatif
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 6. Overtime vs Attrition
        st.subheader("6️⃣ Overtime vs Attrition")
        
        overtime_attrition = pd.crosstab(df['OverTime'], df['Attrition'], normalize='index') * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=overtime_attrition.index,
            y=overtime_attrition['Yes'],
            name='Resign Rate (%)',
            marker_color='#e74c3c',
            text=overtime_attrition['Yes'].round(1),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Attrition Rate: Overtime vs No Overtime',
            xaxis_title='Overtime',
            yaxis_title='Attrition Rate (%)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("""
        ⚠️ **Insight Penting**: 
        Karyawan yang sering OVERTIME memiliki attrition rate lebih tinggi!
        Ini indikasi bahwa work-life balance sangat mempengaruhi keputusan resign.
        """)

# ============================================
# 13. PAGE: MODEL TRAINING
# ============================================

elif page == "🤖 Model Training":
    st.markdown('<h1 class="main-header">🤖 Model Training & Evaluation</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is not None:
        st.success("✅ Dataset berhasil di-load!")
        
        # Tampilkan info dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Target", "Attrition (Yes/No)")
        
        st.markdown("---")
        
        # Button untuk start training
        if st.button("🚀 Start Training Models", type="primary"):
            
            with st.spinner("🔄 Preprocessing data..."):
                # Preprocessing
                df_processed, label_encoders, scaler = preprocess_data(df)
                
                st.success("✅ Preprocessing selesai!")
                
                # Tampilkan hasil encoding
                with st.expander("📋 Lihat Hasil Preprocessing"):
                    st.write("**Encoded Data (5 rows pertama):**")
                    st.dataframe(df_processed.head())
                    
                    st.write("**Label Encoders:**")
                    for col, encoder in label_encoders.items():
                        st.write(f"- **{col}**: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
            
            st.markdown("---")
            
            # Split data
            st.subheader("✂️ Split Data: Train vs Test")
            
            # Separate features and target
            X = df_processed.drop('Attrition', axis=1)
            y = df_processed['Attrition']
            
            # Split dengan ratio 70:30
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.3,  # 30% untuk testing
                random_state=42,  # Seed untuk reproducibility
                stratify=y  # Pastikan proporsi class sama di train dan test
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎓 Training Data", f"{len(X_train)} samples ({len(X_train)/len(df)*100:.0f}%)")
            with col2:
                st.metric("🧪 Testing Data", f"{len(X_test)} samples ({len(X_test)/len(df)*100:.0f}%)")
            
            st.info("""
            ℹ️ **Penjelasan Split:**
            - **Training Set (70%)**: Digunakan untuk "mengajarkan" model
            - **Testing Set (30%)**: Digunakan untuk evaluasi (model belum pernah lihat data ini)
            - **Stratify**: Memastikan proporsi STAY vs RESIGN sama di train dan test
            """)
            
            st.markdown("---")
            
            # Train models
            st.subheader("🎓 Training 5 Machine Learning Models")
            
            models, results = train_models(X_train, X_test, y_train, y_test)
            
            st.success("🎉 Training berhasil!")
            
            # Save to session state
            st.session_state['models'] = models
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = X.columns.tolist()
            st.session_state['label_encoders'] = label_encoders
            st.session_state['scaler'] = scaler
            
            st.markdown("---")
            
            # Show results
            st.subheader("📊 Model Comparison")
            
            results_df = pd.DataFrame(results)
            
            # Format percentage
            for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
                results_df[col] = results_df[col].apply(lambda x: f"{x*100:.2f}%")
            
            # Style dataframe
            st.dataframe(
                results_df.style.set_properties(**{
                    'background-color': '#f0f2f6',
                    'border-color': 'white'
                }),
                use_container_width=True
            )
            
            # Find best model
            results_numeric = pd.DataFrame(results)
            best_model_idx = results_numeric['F1-Score'].idxmax()
            best_model_name = results_numeric.iloc[best_model_idx]['Model']
            best_f1 = results_numeric.iloc[best_model_idx]['F1-Score']
            
            st.success(f"""
            🏆 **Model Terbaik**: {best_model_name}
            - F1-Score: {best_f1*100:.2f}%
            """)
            
            st.markdown("---")
            
            # Visualisasi comparison
            st.subheader("📊 Visual Comparison")
            
            results_numeric_viz = results_numeric.copy()
            results_numeric_viz = results_numeric_viz.set_index('Model')
            
            fig = go.Figure()
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
            
            for metric in metrics:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_numeric_viz.index,
                    y=results_numeric_viz[metric] * 100,
                    text=results_numeric_viz[metric].apply(lambda x: f"{x*100:.1f}%"),
                    textposition='auto'
                ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score (%)',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ROC Curves
            st.subheader("📈 ROC Curves - All Models")
            plot_roc_curves(models, X_test, y_test)
            
            st.markdown("---")
            
            # Detailed evaluation per model
            st.subheader("🔍 Detailed Evaluation per Model")
            
            selected_model = st.selectbox(
                "Pilih model untuk lihat detail:",
                list(models.keys())
            )
            
            if selected_model:
                model = models[selected_model]
                y_pred = model.predict(X_test)
                
                # Confusion Matrix
                st.markdown(f"### Confusion Matrix - {selected_model}")
                plot_confusion_matrix(y_test, y_pred, selected_model)
                
                st.markdown("---")
                
                # Classification Report
                st.markdown(f"### Classification Report - {selected_model}")
                
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
                
                st.markdown("""
                <div class="info-box">
                <b>📊 Penjelasan Classification Report:</b><br>
                • <b>Precision</b>: Dari yang diprediksi positive, berapa yang benar<br>
                • <b>Recall</b>: Dari yang actual positive, berapa yang terdeteksi<br>
                • <b>F1-Score</b>: Harmonic mean dari Precision dan Recall<br>
                • <b>Support</b>: Jumlah actual data di setiap class
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Feature Importance (jika tree-based model)
                if selected_model in ['Decision Tree', 'Random Forest', 'XGBoost']:
                    st.markdown(f"### Feature Importance - {selected_model}")
                    plot_feature_importance(
                        model,
                        st.session_state['feature_names'],
                        selected_model
                    )

# ============================================
# 14. PAGE: PREDICTION (REAL MODEL VERSION)
# ============================================

elif page == "🎯 Prediction":
    st.markdown('<h1 class="main-header">🎯 Employee Attrition Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Cek apakah model sudah di-train
    if 'models' not in st.session_state:
        st.warning("⚠️ Model belum di-train! Silakan training terlebih dahulu di halaman Model Training.")
    else:
        models = st.session_state['models']
        scaler = st.session_state['scaler']
        label_encoders = st.session_state['label_encoders']
        feature_names = st.session_state['feature_names']

        st.success("✅ Model siap digunakan!")

        # Pilih model
        selected_model_name = st.selectbox(
            "🤖 Pilih Model:",
            list(models.keys()),
            index=2
        )

        model = models[selected_model_name]

        st.markdown("---")
        st.subheader("📝 Input Data Karyawan")

        col1, col2 = st.columns(2)

        with col1:
            Age = st.number_input("Age", 18, 65, 30)
            MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000, step=100)
            DistanceFromHome = st.number_input("Distance From Home", 1, 30, 10)
            YearsAtCompany = st.number_input("Years At Company", 0, 40, 5)
            YearsInCurrentRole = st.number_input("Years In Current Role", 0, 20, 3)
            JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
            WorkLifeBalance = st.slider("Work Life Balance", 1, 4, 3)

        with col2:
            Gender = st.selectbox("Gender", ["Male", "Female"])
            OverTime = st.selectbox("Over Time", ["Yes", "No"])
            Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            JobRole = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ])

        st.markdown("---")

        if st.button("🔮 Prediksi Attrition", type="primary"):

            # =========================================
            # BUILD INPUT DATA SESUAI FEATURE TRAINING
            # =========================================
            input_dict = {col: 0 for col in feature_names}

            input_dict.update({
                'Age': Age,
                'MonthlyIncome': MonthlyIncome,
                'DistanceFromHome': DistanceFromHome,
                'YearsAtCompany': YearsAtCompany,
                'YearsInCurrentRole': YearsInCurrentRole,
                'JobSatisfaction': JobSatisfaction,
                'WorkLifeBalance': WorkLifeBalance,
                'Gender': label_encoders['Gender'].transform([Gender])[0],
                'OverTime': label_encoders['OverTime'].transform([OverTime])[0],
                'Department': label_encoders['Department'].transform([Department])[0],
                'JobRole': label_encoders['JobRole'].transform([JobRole])[0],
            })

            input_df = pd.DataFrame([input_dict])
            input_df[feature_names] = scaler.transform(input_df[feature_names])

            # =========================================
            # REAL PREDICTION
            # =========================================
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] * 100

            status = "RESIGN" if prediction == 1 else "STAY"

            if probability >= 70:
                risk = "HIGH"
                color = "#e74c3c"
            elif probability >= 40:
                risk = "MEDIUM"
                color = "#f39c12"
            else:
                risk = "LOW"
                color = "#2ecc71"

            st.markdown("---")
            st.subheader("📊 Hasil Prediksi")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div style='background-color:{color}; padding:2rem; border-radius:0.5rem; text-align:center;'>
                    <h2 style='color:white;margin:0;'>{status}</h2>
                    <p style='color:white;'>Prediction</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style='background-color:#34495e; padding:2rem; border-radius:0.5rem; text-align:center;'>
                    <h2 style='color:white;margin:0;'>{probability:.2f}%</h2>
                    <p style='color:white;'>Confidence</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style='background-color:{color}; padding:2rem; border-radius:0.5rem; text-align:center;'>
                    <h2 style='color:white;margin:0;'>{risk}</h2>
                    <p style='color:white;'>Risk Level</p>
                </div>
                """, unsafe_allow_html=True)

            # =========================================
            # REKOMENDASI
            # =========================================
            st.markdown("---")
            st.subheader("💡 Rekomendasi")

            if status == "RESIGN":
                st.warning("""
                ⚠️ **Karyawan Berisiko Tinggi Resign**
                - Lakukan one-on-one meeting
                - Evaluasi gaji & benefit
                - Kurangi overtime
                - Tingkatkan work-life balance
                """)
            else:
                st.success("""
                ✅ **Karyawan Cenderung Bertahan**
                - Pertahankan engagement
                - Berikan pengembangan karier
                - Monitor kepuasan kerja
                """)

# ============================================
# 15. PAGE: ABOUT
# ============================================

elif page == "ℹ️ About":
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ## 📚 Tentang Proyek Ini
    
    Aplikasi ini dibuat sebagai bagian dari tugas **Fundamental Sains Data** 
    dengan topik **Employee Attrition Prediction** menggunakan **Machine Learning**.
    
    ### 🎯 Tujuan
    
    Membangun sistem prediksi berbasis AI untuk membantu perusahaan:
    - Mengidentifikasi karyawan yang berisiko resign
    - Memahami faktor-faktor yang mempengaruhi attrition
    - Mengambil tindakan preventif untuk retention
    
    ### 🛠️ Teknologi yang Digunakan
    
    | Kategori | Teknologi |
    |----------|-----------|
    | **Frontend** | Streamlit |
    | **Data Processing** | Pandas, NumPy |
    | **Visualization** | Matplotlib, Seaborn, Plotly |
    | **Machine Learning** | Scikit-learn, XGBoost |
    | **Deployment** | Streamlit Cloud |
    
    ### 🤖 Machine Learning Models
    
    1. **Logistic Regression** - Linear classification model
    2. **Decision Tree** - Tree-based model dengan interpretability tinggi
    3. **Random Forest** - Ensemble model dengan multiple trees
    4. **Support Vector Machine** - Kernel-based classification
    5. **XGBoost** - Advanced gradient boosting algorithm
    
    ### 📊 Dataset
    
    - **Sumber**: IBM HR Analytics Employee Attrition & Performance
    - **Platform**: Kaggle
    - **Jumlah Records**: 1,470 employees
    - **Jumlah Features**: 35 features
    - **Target**: Attrition (Yes/No)
    
    ### 📈 Evaluation Metrics
    
    - **Accuracy**: Overall correctness
    - **Precision**: Positive prediction accuracy
    - **Recall**: True positive detection rate
    - **F1-Score**: Harmonic mean of Precision & Recall
    - **AUC**: Area Under ROC Curve
    - **Confusion Matrix**: Detailed prediction breakdown
    
    ### ✨ Key Features
    
    1. 📊 **Interactive Data Exploration**
    2. 🤖 **Multiple Model Comparison**
    3.
    """)