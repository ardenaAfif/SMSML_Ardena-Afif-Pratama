from modelling_tuning import init_mlflow, load_and_split_data, train_and_log_model, PROCESSED_DATA_PATH

if __name__ == "__main__":
    print("--- Memulai Script modelling.py (memanggil modelling_tuning untuk Klasifikasi) ---")
    
    mlflow_initialized_successfully = init_mlflow()
    
    if mlflow_initialized_successfully:
        # Memastikan semua return value dari load_and_split_data ditangkap
        X_train, X_test, y_train, y_test, feature_names, class_labels_int = load_and_split_data(PROCESSED_DATA_PATH)
    
        if X_train is not None and y_train is not None: # Cek y_train juga
            # Memanggil fungsi train_and_log_model (sesuai nama di modelling_tuning.py Anda)
            train_and_log_model(X_train, y_train, X_test, y_test, feature_names, class_labels_int)
    else:
        print("Inisialisasi MLflow gagal. Script modelling.py tidak dapat melanjutkan.")
        
    print("--- Script modelling.py (Klasifikasi) Selesai ---")