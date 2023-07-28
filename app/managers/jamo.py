import typing as T
from pathlib import Path
from functools import lru_cache
import sys
import torch

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from jamo import JAMO

class JamoModelManager():
    def __init__(self, model_path, device="cpu"): 
        self.device = device
        self.init_model(model_path, device)

    def init_model(self, model_path):
        self.model = JAMO.from_pretrained("small", model_path, self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, input: torch.Tensor, max_seq_length: int, input_pos: torch.Tensor, temperature: float, top_k: int, multibatch: bool=False) -> torch.Tensor:
        next_token_idx = 0
        try:
            logits = self.model(input, max_seq_length=max_seq_length, input_pos=input_pos)
            if multibatch:
                logits = logits[:, -1]
            else: 
                logits = logits[0, -1]
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                
                if multibatch:
                    target_value = v[:, [-1]]
                else: 
                    target_value = v[[-1]]
                logits = torch.where(logits < target_value, -float("Inf"), logits)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
        except Exception as e:
            raise e
        return next_token_idx

    def clean_cache(self):
        self.model.reset_cache()


@lru_cache(maxsize=1)
def get_jamo_manager(device):
    manager = JamoModelManager(device)

    return manager