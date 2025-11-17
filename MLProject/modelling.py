"""
Modelling Script untuk Predictive Maintenance
Menggunakan MLflow untuk tracking dan autolog
"""

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend untuk menghindari tkinter error

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(data_dir):
    """
    Load data yang sudah diproses
    
    Parameters:
    -----------
    data_dir : str
        Directory berisi data yang sudah diproses
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Data training dan testing
    """
    print(f"Loading preprocessed data from {data_dir}...")
    
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').squeeze()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').squeeze()
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test, experiment_name="Predictive Maintenance"):
    """
    Train model menggunakan Random Forest dengan MLflow autolog
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    experiment_name : str
        Nama experiment MLflow
    
    Returns:
    --------
    model : sklearn model
        Trained model
    """
    # Set MLflow tracking URI (file-based untuk basic, tidak perlu server)
    # Untuk menggunakan server, ganti dengan: "http://127.0.0.1:5000/"
    tracking_uri = "./mlruns"  # File-based backend store
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Enable autologging untuk metrics
    # Disable model logging dari autolog karena kita akan log manual
    mlflow.sklearn.autolog(
        log_models=False,  # Disable auto model logging
        log_input_examples=False,
        log_model_signatures=True
    )
    
    print("\n=== Training Model ===")
    print(f"MLflow tracking URI: {tracking_uri}")
    print("MLflow autolog enabled (metrics only)")
    
    with mlflow.start_run():
        # Initialize model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        print("Training Random Forest Classifier...")
        model.fit(X_train, y_train)
        
        # Predictions untuk evaluasi
        y_test_pred = model.predict(X_test)
        
        # Print results (hanya untuk informasi, tidak di-log manual)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"\n=== Model Performance ===")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_test_pred))
        
        print("\n=== Confusion Matrix ===")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        
        # Log model secara eksplisit ke artifacts
        # Metrics sudah di-log otomatis oleh autolog
        print("\n" + "="*60)
        print("Logging model ke artifacts...")
        try:
            # Pastikan model di-log sebelum run berakhir
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=None  # Tidak register ke model registry
            )

            # Verifikasi model sudah di-log
            run_id = mlflow.active_run().info.run_id
            artifact_uri = mlflow.active_run().info.artifact_uri
            print(f"✓ Model berhasil di-log!")
            print(f"  Run ID: {run_id}")
            print(f"  Artifact URI: {artifact_uri}")
            print(f"  Model location: {artifact_uri}/model")

            # Cek apakah file benar-benar ada (untuk verifikasi)
            # Get experiment ID
            experiment = mlflow.get_experiment_by_name(experiment_name)
            exp_id = experiment.experiment_id
            artifact_path_local = f"./mlruns/{exp_id}/{run_id}/artifacts/model"

            if os.path.exists(artifact_path_local):
                model_files = os.listdir(artifact_path_local)
                print(f"\n✓ Verifikasi: Files di model/ ditemukan:")
                for file in model_files:
                    print(f"    - {file}")

                # Cek file penting
                required_files = ["MLmodel", "conda.yaml", "requirements.txt", "model.pkl"]
                missing = []
                for req_file in required_files:
                    if req_file not in model_files:
                        missing.append(req_file)

                if missing:
                    print(f"  ⚠ Warning: File berikut belum ada: {missing}")
                else:
                    print(f"  ✓ Semua file penting ada!")
            else:
                print(f"  ⚠ Warning: Local path tidak ditemukan: {artifact_path_local}")
                print(f"    (Path akan dibuat saat run selesai)")

        except Exception as e:
            print(f"✗ Error saat logging model: {e}")
            import traceback
            traceback.print_exc()
            raise

        print("\nModel dan metrics telah di-log!")
        print("  - Metrics: di-log otomatis oleh MLflow autolog")
        print("  - Model: di-log ke folder 'model' di artifacts")
        print("="*60)

        # Untuk melihat hasil, jalankan: mlflow ui --backend-store-uri ./mlruns
        print("\nUntuk melihat hasil di MLflow UI, jalankan:")
        print(f"  mlflow ui --backend-store-uri ./mlruns")
        print(f"  Kemudian buka browser ke: http://localhost:5000")
    
    return model


def save_model_locally(model, output_dir="model"):
    """
    Save model secara lokal
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    output_dir : str
        Directory untuk menyimpan model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved locally to {output_dir}/model.pkl")


if __name__ == "__main__":
    # Konfigurasi
    data_dir = os.path.join(os.path.dirname(__file__), "predictive_maintenance_preprocessing")
    experiment_name = "Predictive Maintenance - Basic"
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data(data_dir)
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test, experiment_name)
    
    # Save model locally
    save_model_locally(model)
    
    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETED!")
    print("=" * 50)



