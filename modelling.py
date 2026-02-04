import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import os
import automate_Zaky as az 

# Set nama eksperimen
mlflow.set_experiment("Bank_Marketing_Artifact_Fix")

def train_model():
    print("🚀 Memulai Training...")
    
    # 1. Panggil fungsi dari automate_Zaky.py
    df = az.load_data()
    if df is None:
        print("❌ Gagal load data. Pastikan folder data_raw/bank.csv ada.")
        return
        
    X_train, X_test, y_train, y_test = az.preprocess_data(df)

    # 2. LOGIKA BARU: Cek Active Run (Solusi Error Double Login)
    # Jika dijalankan lewat 'mlflow run' (GitHub Actions), ini akan True
    if mlflow.active_run():
        print(f"ℹ️ Active Run terdeteksi (ID: {mlflow.active_run().info.run_id}). Menggunakan sesi ini.")
    else:
        # Jika dijalankan manual di laptop, kita start run baru
        print("ℹ️ Tidak ada Active Run. Memulai Run baru secara lokal...")
        mlflow.start_run(run_name="Run_Generate_Artifacts")

    # --- Mulai Proses Training & Logging (Perhatikan indentasi sudah rata kiri) ---
    try:
        # Aktifkan Autolog
        mlflow.sklearn.autolog()

        # Train Model
        print("⚙️ Sedang melatih model...")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Akurasi Model: {acc}")

        # --- [LOG ARTEFAK] ---
        print("📦 Sedang mengemas Artefak (conda.yaml, MLmodel, dll)...")
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",       # Nama folder di DagsHub
            registered_model_name="BankMarketingModel" 
        )
        print("🎉 Selesai! Cek folder 'model' di tab Artifacts DagsHub sekarang.")
        
    except Exception as e:
        print(f"❌ Terjadi error: {e}")
    finally:
        # Menutup run jika kita yang membuatnya secara manual
        # (mlflow run akan menutupnya otomatis, tapi ini good practice)
        mlflow.end_run()

if __name__ == "__main__":
    train_model()
