import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================================================================
# Konfigurasi Halaman & Judul
# =============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Stres Mahasiswa",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Dashboard Analisis Stres Mahasiswa")
st.markdown("""
Aplikasi ini menganalisis faktor-faktor yang memengaruhi tingkat stres mahasiswa menggunakan
model *Machine Learning* (Random Forest dan XGBoost). Gunakan sidebar di sebelah kiri untuk menavigasi halaman.
""")

# =============================================================================
# Fungsi Pemuatan Data (dengan Caching)
# =============================================================================
@st.cache_data  # Cache data agar tidak di-load ulang setiap interaksi
def load_data():
    try:
        df = pd.read_csv('StressLevelDataset.csv')
        return df
    except FileNotFoundError:
        st.error("File 'StressLevelDataset.csv' tidak ditemukan. Harap pastikan file tersebut ada di folder yang sama dengan 'app.py'.")
        return None

df = load_data()

if df is None:
    st.stop()

# =============================================================================
# Persiapan Data (Split)
# =============================================================================
@st.cache_data
def split_data(df):
    X = df.drop('stress_level', axis=1)
    y = df['stress_level']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, X, y

X_train, X_test, y_train, y_test, X, y = split_data(df)

# =============================================================================
# Pelatihan Model (dengan Caching)
# =============================================================================
@st.cache_resource # Cache model yang sudah dilatih
def train_rf(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model

@st.cache_resource
def train_xgb(X_train, y_train):
    model = XGBClassifier(
        n_estimators=50,
        max_depth=1,
        learning_rate=0.0002,
        gamma=0.92,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Latih model
rf_model = train_rf(X_train, y_train)
xgb_model = train_xgb(X_train, y_train)

# =============================================================================
# Navigasi Sidebar
# =============================================================================
st.sidebar.title("Navigasi Halaman")
page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "🏠 Pendahuluan & Data",
        "📊 Eksplorasi Data (EDA)",
        "🤖 Hasil Pemodelan",
        "💡 Faktor Paling Berpengaruh",
        "⚖️ Perbandingan Model"
    ]
)

# =============================================================================
# Halaman 1: Pendahuluan & Data
# =============================================================================
if page == "🏠 Pendahuluan & Data":
    st.header("Dataset Tingkat Stres Mahasiswa")
    st.markdown("Dataset ini berisi berbagai faktor psikologis, fisiologis, akademik, dan sosial yang mungkin memengaruhi tingkat stres mahasiswa.")

    st.subheader("Tinjauan Data Mentah")
    st.dataframe(df.head(10))

    if st.checkbox("Tampilkan statistik deskriptif"):
        st.dataframe(df.describe())

    st.subheader("Pembagian Data Latih & Uji")
    st.markdown("Data dibagi menjadi 80% untuk pelatihan model dan 20% untuk pengujian.")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [len(X_train), len(X_test)],
        labels=['Data Latih (80.0%)', 'Data Uji (20.0%)'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#90CAF9', '#F48FB1']
    )
    ax.set_title('Proporsi Data Latih dan Data Uji')
    st.pyplot(fig)

# =============================================================================
# Halaman 2: Eksplorasi Data (EDA)
# =============================================================================
elif page == "📊 Eksplorasi Data (EDA)":
    st.header("Eksplorasi Data (EDA)")

    # --- 1. Distribusi Fitur ---
    st.subheader("1. Distribusi Frekuensi Setiap Fitur")
    st.markdown("Histogram ini menunjukkan bagaimana sebaran nilai untuk setiap faktor.")
    fig_hist = plt.figure(figsize=(20, 18))
    df.hist(bins=15, ax=fig_hist.gca(), layout=(6, 4))
    plt.suptitle('Distribusi Frekuensi dari Setiap Faktor')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig_hist)

    # --- 2. Korelasi ---
    st.subheader("2. Peta Korelasi (Heatmap)")
    st.markdown("Heatmap menunjukkan seberapa kuat hubungan antar faktor. Nilai mendekati 1 atau -1 menunjukkan korelasi kuat.")

    # Hitung korelasi
    corr = df.corr(numeric_only=True)
    stress_corr = corr[['stress_level']].drop('stress_level').sort_values(by='stress_level', key=abs, ascending=False)

    tab1, tab2 = st.tabs(["Korelasi terhadap Stress Level", "Matriks Korelasi Penuh"])

    with tab1:
        fig_corr_focus, ax = plt.subplots(figsize=(4, 10))
        sns.heatmap(stress_corr, annot=True, cmap='coolwarm', fmt=".2f", center=0, ax=ax)
        ax.set_title('Korelasi Setiap Fitur terhadap Stress Level')
        st.pyplot(fig_corr_focus)

    with tab2:
        fig_corr_full, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", center=0, ax=ax)
        ax.set_title('Heatmap Korelasi Lengkap')
        st.pyplot(fig_corr_full)

    # --- 3. Boxplot Interaktif ---
    st.subheader("3. Distribusi Fitur Berdasarkan Tingkat Stres (Boxplot)")
    st.markdown("Pilih fitur untuk melihat perbandingannya di ketiga tingkat stres (0, 1, 2).")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove('stress_level')
    selected_feature = st.selectbox("Pilih Fitur:", numeric_cols)

    fig_box, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='stress_level', y=selected_feature, ax=ax)
    ax.set_title(f'Distribusi {selected_feature} Berdasarkan Tingkat Stres')
    ax.set_xlabel('Stress Level')
    ax.set_ylabel(selected_feature)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig_box)

# =============================================================================
# Halaman 3: Hasil Pemodelan
# =============================================================================
elif page == "🤖 Hasil Pemodelan":
    st.header("Hasil Evaluasi Model pada Data Uji")
    st.markdown("Berikut adalah performa dari kedua model (Random Forest dan XGBoost) saat diuji pada 20% data yang belum pernah dilihat sebelumnya.")

    # Prediksi
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    # Laporan
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
    report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)

    # Matriks Konfusi
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)

    tab1, tab2 = st.tabs(["🌳 Random Forest", "🚀 XGBoost"])

    with tab1:
        st.subheader("Model: Random Forest")
        st.markdown(f"**Akurasi Global: {accuracy_score(y_test, y_pred_rf):.4f}**")
        
        st.text("Classification Report:")
        st.dataframe(pd.DataFrame(report_rf).transpose())

        st.text("Confusion Matrix:")
        fig_cm_rf, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'], ax=ax)
        ax.set_title('Confusion Matrix - Random Forest')
        ax.set_ylabel('Label Aktual')
        ax.set_xlabel('Label Prediksi')
        st.pyplot(fig_cm_rf)

    with tab2:
        st.subheader("Model: XGBoost")
        st.markdown(f"**Akurasi Global: {accuracy_score(y_test, y_pred_xgb):.4f}**")

        st.text("Classification Report:")
        st.dataframe(pd.DataFrame(report_xgb).transpose())

        st.text("Confusion Matrix:")
        fig_cm_xgb, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'], ax=ax)
        ax.set_title('Confusion Matrix - XGBoost')
        ax.set_ylabel('Label Aktual')
        ax.set_xlabel('Label Prediksi')
        st.pyplot(fig_cm_xgb)

# =============================================================================
# Halaman 4: Faktor Paling Berpengaruh
# =============================================================================
elif page == "💡 Faktor Paling Berpengaruh":
    st.header("Analisis Faktor Paling Berpengaruh (Feature Importance)")
    st.markdown("""
    Model Random Forest dapat memberi tahu kita fitur (faktor) mana yang paling ia anggap penting
    saat membuat keputusan untuk memprediksi tingkat stres.
    """)

    # --- 1. Grafik Feature Importance ---
    feature_importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'Fitur (Faktor Stres)': X.columns,
        'Tingkat Kepentingan': feature_importances
    }).sort_values(by='Tingkat Kepentingan', ascending=False)

    fig_imp, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Tingkat Kepentingan', y='Fitur (Faktor Stres)', data=importance_df, ax=ax)
    ax.set_title('Faktor Stres Paling Berpengaruh (Menurut Random Forest)')
    ax.set_xlabel('Tingkat Kepentingan')
    ax.set_ylabel('Fitur (Faktor Stres)')
    st.pyplot(fig_imp)

    # --- 2. Ringkasan Kategori ---
    st.subheader("Ringkasan Berdasarkan Kategori Faktor")
    st.markdown("""
    Untuk mempermudah, fitur-fitur tersebut dikelompokkan ke dalam kategori yang lebih luas.
    Hasilnya menunjukkan bahwa faktor **Mental/Psikologis** dan **Fisik/Fisiologis**
    adalah dua kelompok faktor paling dominan yang digunakan model untuk memprediksi stres.
    """)

    # Perhitungan kategori (sama seperti di skrip Anda)
    importances = rf_model.feature_importances_
    feature_importance_dict = dict(zip(X.columns, importances))
    categories = {
        'Mental / Psikologis': ['anxiety_level', 'self_esteem', 'depression', 'future_career_concerns', 'peer_pressure', 'bullying', 'social_support', 'teacher_student_relationship'],
        'Fisik / Fisiologis': ['blood_pressure', 'sleep_quality', 'headache', 'breathing_problem'],
        'Lingkungan / Sosial-Ekonomi': ['noise_level', 'living_conditions', 'safety', 'basic_needs', 'extracurricular_activities'],
        'Akademik': ['academic_performance', 'study_load'],
        'Riwayat': ['mental_health_history']
    }
    category_importance = {}
    for category, features in categories.items():
        total = sum(feature_importance_dict.get(feat, 0) for feat in features)
        category_importance[category] = total
    
    total_importance = sum(category_importance.values())
    category_importance_pct = {
        category: (imp / total_importance) * 100 for category, imp in category_importance.items()
    }
    df_category = pd.DataFrame(
        list(category_importance_pct.items()),
        columns=['Kategori', 'Total Feature Importance (%)']
    ).sort_values('Total Feature Importance (%)', ascending=False)
    
    df_category['Total Feature Importance (%)'] = df_category['Total Feature Importance (%)'].round(2)
    
    st.dataframe(df_category, use_container_width=True)


# =============================================================================
# Halaman 5: Perbandingan Model
# =============================================================================
elif page == "⚖️ Perbandingan Model":
    st.header("Perbandingan Kinerja Model: Random Forest vs. XGBoost")
    
    # Ambil laporan dari Halaman 3 (jika belum di-generate)
    report_rf = classification_report(y_test, rf_model.predict(X_test), output_dict=True)
    report_xgb = classification_report(y_test, xgb_model.predict(X_test), output_dict=True)
    cm_rf = confusion_matrix(y_test, rf_model.predict(X_test))
    cm_xgb = confusion_matrix(y_test, xgb_model.predict(X_test))

    # --- 1. Perbandingan Metrik (Precision, Recall, F1) ---
    st.subheader("1. Perbandingan Metrik per Kelas")
    
    classes = ['0', '1', '2']
    metrics = ['precision', 'recall', 'f1-score']
    data = []
    for cls in classes:
        for metric in metrics:
            rf_val = report_rf[cls][metric]
            xgb_val = report_xgb[cls][metric]
            data.append({'Kelas': cls, 'Metrik': metric, 'Random Forest': rf_val, 'XGBoost': xgb_val})
    df_metrics = pd.DataFrame(data)

    fig_metrics, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, metric in enumerate(metrics):
        metric_data = df_metrics[df_metrics['Metrik'] == metric]
        x = np.arange(len(classes))
        width = 0.35

        axes[i].bar(x - width/2, metric_data['Random Forest'], width, label='Random Forest', color='skyblue')
        axes[i].bar(x + width/2, metric_data['XGBoost'], width, label='XGBoost', color='salmon')
        
        axes[i].set_xlabel('Kelas Stress')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'Perbandingan {metric.capitalize()}')
        axes[i].set_xticks(x, classes)
        axes[i].set_ylim(0, 1.05)
        axes[i].legend()

    plt.tight_layout()
    st.pyplot(fig_metrics)

    # --- 2. Tabel Ringkasan ---
    st.subheader("2. Tabel Ringkasan Metrik")
    summary = []
    for cls in classes:
        summary.append({
            'Kelas': cls,
            'RF Precision': f"{report_rf[cls]['precision']:.3f}",
            'XGB Precision': f"{report_xgb[cls]['precision']:.3f}",
            'RF Recall': f"{report_rf[cls]['recall']:.3f}",
            'XGB Recall': f"{report_xgb[cls]['recall']:.3f}",
            'RF F1': f"{report_rf[cls]['f1-score']:.3f}",
            'XGB F1': f"{report_xgb[cls]['f1-score']:.3f}",
        })
    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df.set_index('Kelas'), use_container_width=True)

    # --- 3. Perbandingan Confusion Matrix ---
    st.subheader("3. Perbandingan Confusion Matrix")
    fig_cm, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0','1','2'], yticklabels=['0','1','2'], ax=axes[0])
    axes[0].set_title(f'Random Forest\nAkurasi: {report_rf["accuracy"]:.4f}')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['0','1','2'], yticklabels=['0','1','2'], ax=axes[1])
    axes[1].set_title(f'XGBoost\nAkurasi: {report_xgb["accuracy"]:.4f}')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    st.pyplot(fig_cm)

    # --- 4. Analisis Teks ---
    st.subheader("4. Analisis Perbandingan")
    st.markdown("""
    (Analisis ini diambil langsung dari skrip asli Anda)
    
    Meskipun akurasi total kedua model identik (**90.91%**), terdapat perbedaan nuansa yang signifikan.
    
    * **Random Forest**: Tampak lebih seimbang dan stabil secara keseluruhan. Ia lebih baik dalam *recall* (menemukan) kasus stres sedang (Kelas 1).
    * **XGBoost**: Menunjukkan keunggulan spesifik dalam **presisi kelas 1 (100%)**, yang berarti jika XGBoost memprediksi "stres sedang", prediksinya *selalu benar* (tidak ada *false positive* untuk kelas 1).
    
    **Kesimpulan:**
    Pilihan model sangat bergantung pada prioritas aplikasi.
    
    1.  Jika tujuan utamanya adalah **menghindari kesalahan klasifikasi positif** (misalnya, salah mendiagnosa seseorang sebagai stres sedang), **XGBoost** adalah pilihan yang lebih superior.
    2.  Jika tujuannya adalah **memastikan tidak ada kasus stres sedang yang terlewatkan**, **Random Forest** lebih diutamakan.
    """)