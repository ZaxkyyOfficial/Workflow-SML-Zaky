import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Untuk menyimpan model
import mlflow
import dagshub
import automate_Zaky as az  # Script preprocessing kamu

# --- SETUP DAGSHUB & MLFLOW ---
dagshub.init(repo_owner='arifiyantobahtyar', repo_name='Eksperimen_SML_Zaky', mlflow=True)
mlflow.set_experiment("Bank_Marketing_Simple_Train")

def train_simple_model():
    print("Memulai Training Sederhana...")
    
    # 1. Load Data
    df = az.load_data()
    X_train, X_test, y_train, y_test = az.preprocess_data(df)

    # 2. Cek jumlah fitur (PENTING untuk Serving nanti)
    print(f"Model mengharapkan {X_train.shape[1]} fitur input.")

    # 3. Mulai MLflow Run (Autolog)
    mlflow.sklearn.autolog() # Fitur otomatis mencatat parameter & metrik
    
    with mlflow.start_run(run_name="Simple_RandomForest"):
        # Train Model Sederhana
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {acc}")

        # 4. SIMPAN MODEL KE FILE (Wajib untuk Kriteria 4)
        joblib.dump(model, "model_bank.pkl")
        print("Model berhasil disimpan sebagai 'model_bank.pkl'")

if __name__ == "__main__":
    train_simple_model()