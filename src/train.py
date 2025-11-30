import os
import joblib
import pandas as pd
from ruamel.yaml import YAML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import mlflow

from src.common.settings import settings


def main():
    yaml = YAML()
    with open(settings.params_file, "r") as f:
        params = yaml.load(f)

    random_state = params["train"]["random_state"]

    root_dir      = settings.root_dir
    data_dir      = f"{root_dir}/data"
    processed_dir = f"{data_dir}/processed"
    X_train       = pd.read_csv(f"{processed_dir}/X_train.csv")
    X_test        = pd.read_csv(f"{processed_dir}/X_test.csv")
    y_train       = pd.read_csv(f"{processed_dir}/y_train.csv")
    y_test        = pd.read_csv(f"{processed_dir}/y_test.csv")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

    print("Training Logistic Regression model...\n")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    with mlflow.start_run(run_name="wine_quality"):
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model", "Logistic Regression")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        mlflow.log_metric("balanced_accuracy", balanced_acc)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    models_dir = f"{root_dir}/models"
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(model, f"{models_dir}/best_model.pkl")
    joblib.dump(scaler, f"{models_dir}/scaler.pkl")

    print(f"\nModel and scaler saved to {models_dir}/")

if __name__ == "__main__":
    main()
    