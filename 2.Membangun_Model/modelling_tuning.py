import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
)
import mlflow
import dagshub # Tetap ada untuk opsi DagsHub
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

DAGSHUB_USERNAME = "ardenaAfif" 
DAGSHUB_REPO_NAME = "SMSML_Ardena-Afif"
MLFLOW_EXPERIMENT_NAME = "Shopping_Trends_Classification"

TRACKING_MODE_PREFERENCE = "dagshub"
LOCAL_MLFLOW_SERVER_URI = "http://127.0.0.1:5000"

PROCESSED_DATA_PATH = "processed_shopping_trends.csv"

os.makedirs("artifacts_temp", exist_ok=True)

def init_mlflow():
    global MLFLOW_EXPERIMENT_NAME
    
    if TRACKING_MODE_PREFERENCE == "dagshub":
        try:
            print("Mencoba inisialisasi DagsHub...")
            dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
            mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")
            print(f"MLflow tracking URI diatur ke DagsHub: {mlflow.get_tracking_uri()}")
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            print(f"Eksperimen MLflow diatur ke: {MLFLOW_EXPERIMENT_NAME}")
            return True
        except Exception as e:
            print(f"Error saat inisialisasi DagsHub: {e}. Beralih ke opsi tracking lain.")

    if TRACKING_MODE_PREFERENCE == "local_server" or \
       (TRACKING_MODE_PREFERENCE == "dagshub" and mlflow.get_tracking_uri() is None):
        try:
            print(f"Mencoba menghubungkan ke Local MLflow Server di: {LOCAL_MLFLOW_SERVER_URI}")
            mlflow.set_tracking_uri(LOCAL_MLFLOW_SERVER_URI)
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            if not experiment:
                print(f"Eksperimen '{MLFLOW_EXPERIMENT_NAME}' tidak ditemukan di server lokal, membuat baru...")
                mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            print(f"MLflow tracking URI diatur ke Local Server: {mlflow.get_tracking_uri()}")
            print(f"Eksperimen MLflow diatur ke: {MLFLOW_EXPERIMENT_NAME}")
            return True
        except Exception as e:
            print(f"Error saat menghubungkan ke Local MLflow Server ({LOCAL_MLFLOW_SERVER_URI}): {e}")
            print("Beralih ke local file-based tracking (folder mlruns).")

    # Fallback ke Local File-based Tracking
    try:
        print("Menggunakan local file-based tracking (folder mlruns).")
        # Jika URI sebelumnya (DagsHub/server) gagal dan masih terset, reset ke default file-based
        current_uri = mlflow.get_tracking_uri()
        if current_uri and ("dagshub.com" in current_uri or LOCAL_MLFLOW_SERVER_URI in current_uri):
            print(f"Resetting tracking URI from {current_uri} to default file-based.")
            mlflow.set_tracking_uri(None) # Reset URI untuk menggunakan ./mlruns

        MLFLOW_EXPERIMENT_NAME_LOCAL = MLFLOW_EXPERIMENT_NAME + "_localfiles"
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME_LOCAL)
        print(f"MLflow tracking URI diatur ke local files: {mlflow.get_tracking_uri()} (default: ./mlruns)")
        print(f"Eksperimen MLflow diatur ke: {MLFLOW_EXPERIMENT_NAME_LOCAL}")
        return True
    except Exception as e:
        print(f"Error saat mengatur fallback local file tracking: {e}")
        return False
    
def load_and_split_data(data_path):
    """Memuat data yang sudah diproses dan membaginya untuk klasifikasi."""
    print(f"Memuat data dari: {data_path}")
    df = pd.read_csv(data_path)
    print("Data berhasil dimuat.")
    print(f"Shape df awal: {df.shape}")
    print(f"Kolom df (5 pertama): {df.columns.tolist()[:5]}, (5 terakhir): {df.columns.tolist()[-5:]}")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1] 
    
    print(f"Shape X (fitur): {X.shape}, Shape y (target): {y.shape}")
    target_column_name = df.columns[-1]
    print(f"Nama kolom target yang dipilih (dari iloc): {target_column_name}")
    
    if y.ndim != 1:
        raise ValueError(f"Target y harus 1D, tetapi shape-nya adalah {y.shape}. Pastikan kolom terakhir adalah target '{target_column_name}' yang sudah di-labelencode.")

    # Menggunakan kembali stratify=y untuk klasifikasi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data dibagi menjadi: {len(X_train)} train, {len(X_test)} test.")
    
    class_labels_int = sorted(np.unique(y_train).astype(int).tolist()) # Pastikan integer dan urut
    print(f"Label kelas unik (integer) dari y_train: {class_labels_int}")
    return X_train, X_test, y_train, y_test, X.columns.tolist(), class_labels_int

def plot_confusion_matrix(y_true, y_pred, class_names_str, run_id):
    """Membuat dan menyimpan plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[int(s) for s in class_names_str]) # Pastikan labels sesuai urutan class_names_str
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_str)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names_str)), max(5, len(class_names_str) * 0.8)))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - Run {run_id[:8]}")
    plot_path = f"artifacts_temp/confusion_matrix_run_{run_id[:8]}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path

def log_classification_report_as_json(y_true, y_pred, target_names_str, run_id):
    """Membuat, menyimpan, dan me-log classification report sebagai file JSON."""
    report = classification_report(y_true, y_pred, target_names=target_names_str, output_dict=True, zero_division=0)
    report_path = f"artifacts_temp/classification_report_run_{run_id[:8]}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    return report_path

def train_and_log_model(X_train, y_train, X_test, y_test, feature_names, class_labels_int):
    """Melatih model klasifikasi, melakukan hyperparameter tuning, dan log ke MLflow."""
    
    class_labels_str = [str(label) for label in class_labels_int]
    print(f"Label kelas yang digunakan untuk plot/report (string): {class_labels_str}")

    param_grid = { # Nama variabel disesuaikan dengan yang Anda berikan
        'n_estimators': [50, 100], 
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    grid_search = GridSearchCV( # Nama variabel disesuaikan
        RandomForestClassifier(random_state=42),
        param_grid, # Menggunakan param_grid yang didefinisikan di atas
        cv=3,
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1
    )

    print("Memulai Hyperparameter Tuning dengan GridSearchCV...")
    grid_search.fit(X_train, y_train)
    print("Hyperparameter Tuning selesai.")

    best_params = grid_search.best_params_ # Nama variabel disesuaikan
    best_model = grid_search.best_estimator_ # Nama variabel disesuaikan
    
    print(f"Parameter terbaik ditemukan: {best_params}")

    with mlflow.start_run(run_name="RandomForest_Tuned_Best_Classifier") as parent_run: # Nama run disesuaikan
        parent_run_id = parent_run.info.run_id
        print(f"MLflow Parent Run ID (Best Classifier): {parent_run_id}")
        
        active_uri = mlflow.get_tracking_uri()
        if active_uri and ("dagshub.com" in active_uri or LOCAL_MLFLOW_SERVER_URI in active_uri) :
             print(f"Lihat run di UI MLflow: {active_uri}/#/experiments/{parent_run.info.experiment_id}/runs/{parent_run_id}")
        else:
             print(f"Run disimpan di: {active_uri}. Jalankan 'mlflow ui' untuk melihat (jika di folder mlruns).")

        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", grid_search.cv)
        mlflow.log_param("scoring_metric", grid_search.scoring)

        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        y_proba_test = best_model.predict_proba(X_test)

        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_macro_train = f1_score(y_train, y_pred_train, average='macro', zero_division=0)
        f1_macro_test = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        precision_macro_test = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
        recall_macro_test = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
        
        if len(class_labels_int) > 1 :
            # Menggunakan class_labels_int untuk parameter labels di roc_auc_score
            roc_auc_ovr_test = roc_auc_score(y_test, y_proba_test, multi_class='ovr', average='macro', labels=class_labels_int)
            print(f"ROC AUC OvR Macro (Test): {roc_auc_ovr_test:.4f}")
            mlflow.log_metric("roc_auc_ovr_macro_test", roc_auc_ovr_test)
        else:
            print("ROC AUC tidak dihitung karena jumlah kelas <= 1.")

        print(f"Accuracy (Train): {accuracy_train:.4f}")
        print(f"Accuracy (Test): {accuracy_test:.4f}")
        # ... (print metrik lainnya)

        mlflow.log_metric("accuracy_train", accuracy_train)
        mlflow.log_metric("accuracy_test", accuracy_test)
        mlflow.log_metric("f1_macro_train", f1_macro_train)
        mlflow.log_metric("f1_macro_test", f1_macro_test)
        mlflow.log_metric("precision_macro_test", precision_macro_test)
        mlflow.log_metric("recall_macro_test", recall_macro_test)
        
        cm_plot_path = plot_confusion_matrix(y_test, y_pred_test, class_labels_str, parent_run_id)
        mlflow.log_artifact(cm_plot_path, "plots")
        print(f"Confusion matrix plot dilog sebagai artefak: {cm_plot_path}")

        report_json_path = log_classification_report_as_json(y_test, y_pred_test, class_labels_str, parent_run_id)
        mlflow.log_artifact(report_json_path, "reports")
        print(f"Classification report (JSON) dilog sebagai artefak: {report_json_path}")

        mlflow.sklearn.log_model(
            sk_model=best_model, # Menggunakan best_model
            artifact_path="random-forest-best-classifier", # Nama path artefak disesuaikan
            input_example=X_train.iloc[[0]],
            # Nama model yang diregistrasi disesuaikan
            registered_model_name=f"{MLFLOW_EXPERIMENT_NAME}-RF-Best-Classifier" 
        )
        print("Model Classifier terbaik berhasil dilog ke MLflow.")

        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importances_df = pd.DataFrame(
                {'feature': feature_names, 'importance': importances}
            ).sort_values(by='importance', ascending=False)
            
            plt.figure(figsize=(10, max(6, len(feature_names) // 3)))
            sns.barplot(x='importance', y='feature', data=feature_importances_df.head(20))
            plt.title(f'Top 20 Feature Importances (Classifier) - Run {parent_run_id[:8]}')
            plt.tight_layout()
            fi_plot_path = f"artifacts_temp/feature_importances_classifier_run_{parent_run_id[:8]}.png" # Nama file disesuaikan
            plt.savefig(fi_plot_path)
            plt.close()
            mlflow.log_artifact(fi_plot_path, "plots")
            print(f"Feature importances plot (Classifier) dilog sebagai artefak: {fi_plot_path}")
            
            fi_csv_path = f"artifacts_temp/feature_importances_classifier_run_{parent_run_id[:8]}.csv" # Nama file disesuaikan
            feature_importances_df.to_csv(fi_csv_path, index=False)
            mlflow.log_artifact(fi_csv_path, "reports")
            print(f"Feature importances (CSV Classifier) dilog sebagai artefak: {fi_csv_path}")

        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt", "environment")
            print("requirements.txt dilog sebagai artefak.")

        print(f"\nEksperimen model classifier terbaik selesai. Run ID: {parent_run_id}")

if __name__ == "__main__":
    print("--- Memulai Script Pelatihan Model Klasifikasi & Tuning ---")
    mlflow_initialized_successfully = init_mlflow()
    
    if mlflow_initialized_successfully:
        # Memastikan semua return value dari load_and_split_data ditangkap
        X_train, X_test, y_train, y_test, feature_names, class_labels_int = load_and_split_data(PROCESSED_DATA_PATH)
        
        if X_train is not None and y_train is not None: # Cek y_train juga
            # Memanggil fungsi train_and_log_model (sesuai nama di script Anda)
            train_and_log_model(X_train, y_train, X_test, y_test, feature_names, class_labels_int)
    else:
        print("Inisialisasi MLflow gagal. Script tidak dapat melanjutkan.")
    
    print("--- Script Klasifikasi Selesai ---")