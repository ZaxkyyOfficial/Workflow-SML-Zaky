import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import os
import automate_Zaky as az # <--- Ini memanggil file automate kamu yang sudah benar tadi

# Set nama eksperimen
mlflow.set_experiment("Bank_Marketing_Artifact_Fix")

def train_model():
    print("🚀 Memulai Training...")
    
    # 1. Panggil fungsi dari automate_Zaky.py
    # Karena automate kamu membaca file lokal, pastikan folder data_raw dan bank.csv ada
    df = az.load_data()
    if df is None:
        print("❌ Gagal load data. Pastikan folder data_raw/bank.csv ada.")
        return
        
    X_train, X_test, y_train, y_test = az.preprocess_data(df)

    # 2. Mulai MLflow Run (DENGAN ARTEFAK LENGKAP)
    with mlflow.start_run(run_name="Run_Generate_Artifacts"):
        
        # Aktifkan Autolog (untuk grafik)
        mlflow.sklearn.autolog()

        # Train Model
        print("⚙️ Sedang melatih model...")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Akurasi Model: {acc}")

        # --- [INI KUNCINYA AGAR ARTEFAK MUNCUL] ---
        print("📦 Sedang mengemas Artefak (conda.yaml, MLmodel, dll)...")
        
        # Perintah ini memaksa MLflow membuat folder 'model' berisi semua file yang diminta reviewer
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",       # Nama folder di DagsHub nanti
            registered_model_name="BankMarketingModel" 
        )
        print("🎉 Selesai! Cek folder 'model' di tab Artifacts DagsHub sekarang.")

if __name__ == "__main__":
    train_model()