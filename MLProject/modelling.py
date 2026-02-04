import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import os
import automate_Zaky as az 

def train_model():
    print("🚀 Memulai Training...")

    # --- LOGIKA BARU: DETEKSI ENVIRONMENT ---
    # Cek apakah script ini dijalankan oleh 'mlflow run' (GitHub Actions)
    # Jika ya, environment variable MLFLOW_RUN_ID pasti ada.
    in_mlflow_run = os.environ.get("MLFLOW_RUN_ID") is not None

    if in_mlflow_run:
        print("ℹ️ Terdeteksi berjalan via MLflow Run (CI/CD).")
        # JANGAN set_experiment di sini, karena akan bentrok dengan Run ID dari GitHub
    else:
        print("ℹ️ Terdeteksi berjalan Manual (Lokal).")
        # Hanya set experiment jika jalan di laptop sendiri
        mlflow.set_experiment("Bank_Marketing_Artifact_Fix")

    # 1. Panggil fungsi dari automate_Zaky.py
    df = az.load_data()
    if df is None:
        print("❌ Gagal load data. Pastikan folder data_raw/bank.csv ada.")
        return
        
    X_train, X_test, y_train, y_test = az.preprocess_data(df)

    # 2. KONFIGURASI START RUN
    # Jika di CI/CD: start_run() kosong akan otomatis 'nempel' ke Run ID yang sudah dibuat GitHub
    # Jika di Lokal: kita beri nama run baru
    run_context = mlflow.start_run() if in_mlflow_run else mlflow.start_run(run_name="Run_Generate_Artifacts")

    with run_context:
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
            print("📦 Sedang mengemas Artefak...")
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",       
                registered_model_name="BankMarketingModel" 
            )
            print("🎉 Selesai! Cek DagsHub sekarang.")
            
        except Exception as e:
            print(f"❌ Terjadi error saat logging: {e}")
            raise e

if __name__ == "__main__":
    train_model()
