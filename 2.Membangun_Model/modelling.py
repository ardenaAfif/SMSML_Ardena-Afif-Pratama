import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Aktifkan autolog dari MLflow
mlflow.autolog()

# Path ke data hasil preprocessing
PROCESSED_DATA_PATH = "processed_shopping_trends.csv"

def load_and_split_data(path):
    try:
        df = pd.read_csv(path)
        print("Data berhasil dibaca.")

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        print(f"Kolom target: {df.columns[-1]}")

        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return None, None, None, None

if __name__ == "__main__":
    print("--- Memulai Script modelling.py ---")

    X_train, X_test, y_train, y_test = load_and_split_data(PROCESSED_DATA_PATH)

    if X_train is not None:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Akurasi: {acc}")
    
    print("--- Script modelling.py Selesai ---")