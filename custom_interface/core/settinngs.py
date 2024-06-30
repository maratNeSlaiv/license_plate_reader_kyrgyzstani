from pathlib import Path
import os
from typing import Any

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()

class Settings(BaseSettings):
    BASE_DIR: Any = Path(__file__).resolve().parent.parent.parent
    DEBUG: bool = True
    HOST: str = '0.0.0.0'
    PORT: int = 8070
    RELOAD: bool = True
    IS_PROD: bool = False
    API_V1_STR: str = '/api/v1'
    MEDIA_FOLDER: Any = BASE_DIR / 'media'
    RABBIT_EXCHANGE: str = 'barriers'
    REDIS_URL: str = 'redis://localhost'
    RECORDING_URL: str = 'http://localhost:8060'
    BACK_END_URL: str = 'http://localhost:8080'
    RABBIT_HOST: str = '127.0.0.1'

    # db config
    PG_CONF: str = f"dbname={os.getenv('POSTGRES_DB')} \
        user={os.getenv('POSTGRES_USER')} password={os.getenv('POSTGRES_PASSWORD')}\
              host={os.getenv('POSTGRES_HOST')} port={os.getenv('POSTGRES_PORT')}"

settings = Settings()