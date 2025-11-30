import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split

from src.common.settings import settings


def main():
    yaml = YAML()
    with open(settings.params_file, "r") as f:
        params = yaml.load(f)

    random_state = params["prepare"]["random_state"]
    split_ratio = params["prepare"]["split_ratio"]

    root_dir = settings.root_dir
    data_dir = f"{root_dir}/data"
    raw_data_filename = f"{data_dir}/raw/{settings.dataset_file}"

    df = pd.read_csv(raw_data_filename)

    # Проверка на пропущенные значения
    print(f"df shape: {df.shape}")
    print("Missing values per column:")
    print(df.isnull().sum())
    print(f"\nTotal rows: {len(df)}")

    # Удаление дубликатов
    df_clean = df.drop_duplicates()
    print(f"Rows after removing duplicates: {len(df_clean)}")

    # Удаление ID
    if 'Id' in df_clean.columns:
        df_clean = df_clean.drop('Id', axis=1)

    # Разделение на признаки (X) и целевую переменную (y)
    X = df_clean.drop('quality', axis=1)
    y = df_clean['quality']

    # Проверка распределения классов
    print("\nClass distribution:")
    class_counts = y.value_counts().sort_index()
    print(f"Class counts:\n{class_counts}")
    # Расчет процентов по каждому классу
    class_percents = class_counts / len(y) * 100
    print(f"Class percentages:\n{class_percents.round(2)}")

    # Разделение на train/test (80/20 split, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=split_ratio, 
        random_state=random_state,
        stratify=y
    )

    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Сохранение подготовленных данных
    processed_dir = f"{data_dir}/processed"
    os.makedirs(processed_dir, exist_ok=True)

    X_train.to_csv(f"{processed_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{processed_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{processed_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{processed_dir}/y_test.csv", index=False)

    print(f"\nData saved to {processed_dir}/")
    print("Files: X_train.csv, X_test.csv, y_train.csv, y_test.csv")


if __name__ == "__main__":
    main()
