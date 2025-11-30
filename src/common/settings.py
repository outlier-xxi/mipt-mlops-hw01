from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    root_dir    : str = "/study/ml-ops/ml-ops-prod/homework/mlops-hw01"
    dataset_file: str = "WineQT.csv"
    params_file: str = "params.yaml"


settings = Settings()