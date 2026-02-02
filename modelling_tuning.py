import mlflow
import dagshub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import automate_Zaky as az
import os

# --- BAGIAN INI SUDAH SAYA PERBAIKI ---
# Saya ganti 'USERNAME_DAGSHUB_KAMU' menjadi 'arifiyantobahtyar' sesuai log kamu
DAGSHUB_REPO_OWNER = "arifiyantobahtyar"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Zaky"

try:
    # Setup DagsHub Tracking
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
except Exception as e:
    print(f"Error init DagsHub: {e}")

def train_and_evaluate():
    print("Memulai proses training...")

    # 1. Load & Preprocess Data
    df = az.load_data()
    if df is None:
        print("Gagal load data. Training dibatalkan.")
        return
    X_train, X_test, y_train, y_test = az.preprocess_data(df)

    # 2. Definisi Hyperparameter
    param_grid = [
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': 20}
    ]

    # Set Experiment Name
    mlflow.set_experiment("Bank_Marketing_Experiment_Zaky")

    for params in param_grid:
        with mlflow.start_run(run_name=f"RF_Depth_{params['max_depth']}"):
            print(f"Training dengan params: {params}")

            # A. Init & Train Model
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=42
            )
            model.fit(X_train, y_train)

            # B. Prediksi
            y_pred = model.predict(X_test)

            # C. Hitung Metrik
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"Result -> Acc: {acc:.4f}, F1: {f1:.4f}")

            # D. Logging ke MLflow
            mlflow.log_param("n_estimators", params['n_estimators'])
            mlflow.log_param("max_depth", params['max_depth'])
            mlflow.log_param("model_type", "RandomForest")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, "model")

            # E. Membuat & Log Artefak
            # Artefak 1: Confusion Matrix
            plt.figure(figsize=(6, 5))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix (Depth={params["max_depth"]})')
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()

            # Artefak 2: Feature Importance
            plt.figure(figsize=(10, 6))
            feat_importances = pd.Series(model.feature_importances_, index=pd.DataFrame(X_train).columns)
            feat_importances.nlargest(10).plot(kind='barh')
            plt.title('Top 10 Feature Importances')
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()

    print("Training Selesai. Cek DagsHub kamu!")

if __name__ == "__main__":
    train_and_evaluate()
