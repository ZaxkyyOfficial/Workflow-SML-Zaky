import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data():
    """
    Fungsi untuk memuat dataset Bank Marketing dari UCI.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    print("Mengunduh dataset...")
    try:
        response = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall()
        # Membaca file (separator adalah titik koma)
        df = pd.read_csv('bank-additional/bank-additional-full.csv', sep=';')
        print("Dataset berhasil dimuat.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Fungsi untuk membersihkan data, encoding, dan scaling.
    Mengembalikan X_train, X_test, y_train, y_test yang siap latih.
    """
    # 1. Handling Duplicates
    df = df.drop_duplicates()
    
    # 2. Handling 'unknown' -> NaN
    df.replace('unknown', np.nan, inplace=True)
    
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Pipeline Preprocessing
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
    
    # Agar output tetap pandas DataFrame
    preprocessor.set_output(transform="pandas")
    
    # Fit & Transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("Preprocessing selesai.")
    print(f"Shape Train: {X_train_processed.shape}, Shape Test: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    # Blok ini hanya berjalan jika file dieksekusi langsung (python automate_Zaky.py)
    df = load_data()
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        print("Automasi berhasil!")
