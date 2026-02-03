import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import os
import automate_Zaky as az

# --- BAGIAN PENTING: Setup MLflow Tanpa Login Browser ---
# Kita ambil alamat server langsung dari "Secrets" yang sudah diset di GitHub
uri = os.environ.get("MLFLOW_TRACKING_URI")
if uri:
    mlflow.set_tracking_uri(uri)
    print(f"MLflow Tracking URI diset ke: {uri}")
else:
    print("PERINGATAN: MLFLOW_TRACKING_URI tidak ditemukan di Environment!")

# Set nama eksperimen
mlflow.set_experiment("Bank_Marketing_Simple_Train")

def train_simple_model():
    print("Memulai Training Sederhana...")
    
    # 1. Load Data
    try:
        df = az.load_data()
        X_train, X_test, y_train, y_test = az.preprocess_data(df)
    except Exception as e:
        print(f"Error saat load data: {e}")
        return

    # 2. Mulai MLflow Run
    # Gunakan environment variable MLFLOW_TRACKING_USERNAME & PASSWORD otomatis
    print("Mulai logging ke DagsHub...")
    
    with mlflow.start_run(run_name="CI_Pipeline_Run"):
        # Auto-log parameter & metrics
        mlflow.sklearn.autolog()

        # Train Model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {acc}")

        # 3. Simpan Model Lokal (Untuk Artifact)
        joblib.dump(model, "model_bank.pkl")
        print("Model berhasil disimpan sebagai 'model_bank.pkl'")
        
        # Log model ke MLflow (Opsional, biar muncul di tab Models DagsHub)
        mlflow.sklearn.log_model(model, "model_random_forest")

if __name__ == "__main__":
    train_simple_model()
