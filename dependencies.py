from functools import lru_cache
from fastapi.logger import logger
import torch
from .settings import get_settings


logger.info("Init the Project Dependencies")
env = get_settings()
torch.set_grad_enabled(False)
logger.info("model loaded Start ")
logger.info("Loading Completed!!")
