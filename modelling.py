import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import os  # <--- PENTING: Tambahkan import os
import automate_Zaky as az

# --- BAGIAN INI YANG DIUBAH (HAPUS dagshub.init) ---
# Kita gunakan os.environ agar tidak minta login browser
try:
    print("Mencoba setup MLflow dari Environment Variable...")
    # Mengambil URI dari Secrets GitHub
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print("MLflow Tracking URI berhasil diset!")
    else:
        print("Warning: MLFLOW_TRACKING_URI tidak ditemukan. Cek Secrets GitHub.")
except Exception as e:
    print(f"Error setup MLflow: {e}")

# Set Experiment Name
mlflow.set_experiment("Bank_Marketing_Simple_Train")

# --- BAWAHNYA TETAP SAMA ---
def train_simple_model():
    print("Memulai Training Sederhana...")
    
    # 1. Load Data
    df = az.load_data()
    X_train, X_test, y_train, y_test = az.preprocess_data(df)

    # 2. Cek jumlah fitur
    print(f"Model mengharapkan {X_train.shape[1]} fitur input.")

    # 3. Mulai MLflow Run
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Simple_RandomForest_CI"):
        # Train Model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {acc}")

        # 4. SIMPAN MODEL
        joblib.dump(model, "model_bank.pkl")
        print("Model berhasil disimpan sebagai 'model_bank.pkl'")

if __name__ == "__main__":
    train_simple_model()
