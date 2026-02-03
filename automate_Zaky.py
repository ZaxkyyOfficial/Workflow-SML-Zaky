import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data():
    """
    Fungsi untuk memuat dataset dari folder lokal data_raw.
    """
    # Sesuaikan nama file jika yang kamu punya adalah bank-additional-full.csv
    path = 'data_raw/bank.csv' 
    
    print(f"Mencoba memuat data dari {path}...")
    if os.path.exists(path):
        # Separator bank.csv biasanya titik koma (;)
        df = pd.read_csv(path, sep=';') 
        print("Dataset berhasil dimuat.")
        return df
    else:
        print(f"ERROR: File tidak ditemukan di {path}")
        print("Pastikan folder 'data_raw' ada dan berisi file csv.")
        return None

def preprocess_data(df):
    """
    Membersihkan data, MENYIMPAN HASIL CLEANING, dan melakukan scaling/encoding.
    """
    print("Memulai Preprocessing...")

    # 1. Handling Duplicates
    df = df.drop_duplicates()
    
    # 2. Handling 'unknown' -> NaN
    df.replace('unknown', np.nan, inplace=True)
    
    # --- [BAGIAN WAJIB DARI REVIEWER: SIMPAN DATA BERSIH] ---
    # Kita simpan data yang sudah bersih (no duplicates, handled unknown) sebelum di-split
    os.makedirs('preprocessing', exist_ok=True)
    clean_path = 'preprocessing/clean_data.csv'
    df.to_csv(clean_path, index=False)
    print(f"Data bersih berhasil disimpan di: {clean_path}")
    # --------------------------------------------------------

    # 3. Pisahkan Fitur dan Target
    target_col = 'y'
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan.")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 4. Encoding Target (No=0, Yes=1)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # 5. Split Data (Stratified)
    # Penting: Split dulu baru scaling/encoding untuk mencegah Data Leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Pipeline Preprocessing (Standard Industri)
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    # Agar output tetap pandas DataFrame (memudahkan debug)
    preprocessor.set_output(transform="pandas")
    
    # Fit & Transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("Preprocessing selesai.")
    print(f"Shape Train: {X_train_processed.shape}, Shape Test: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    # Blok ini dijalankan jika file dieksekusi langsung
    df = load_data()
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        print("Automasi Preprocessing Berhasil!")