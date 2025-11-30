from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # root_dir     : str = "/study/ml-ops/ml-ops-prod/homework/mlops-hw01"
    root_dir     : str = str(Path(__file__).parent.parent.parent.resolve())
    dataset_file: str  = "WineQT.csv"
    params_file : str  = "params.yaml"
    tracking_uri: str  = "http://localhost:5000"


settings = Settings()
