import typing as T
from functools import lru_cache
from pydantic import BaseSettings


class ModelSettings(BaseSettings):
    MODEL_NAME = "A"
    JAMO_MODEL_PATH = "model_store/production_A.tar"
    JAMO_MODEL_SIZE = "small"
    BLOCK_SIZE = 256

class MicroBatchSettings(BaseSettings):
    MB_BATCH_SIZE = 64
    MB_MAX_LATENCY = 0.2 
    MB_WORKER_NUM = 1


class DeviceSettings(BaseSettings):
    DEVICE = "cpu"


class Settings(
    ModelSettings,
    MicroBatchSettings,
    DeviceSettings,
):
    CORS_ALLOW_ORIGINS: T.List[str] = [
        "*",
    ]


@lru_cache()
def get_settings():
    setting = Settings()
    return setting