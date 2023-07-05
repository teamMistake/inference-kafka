import typing as T
from pathlib import Path
import sys
import torch

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from settings import get_settings
from managers import (
    get_jamo_manager
)

env = get_settings()


class JamoService:
    def __init__(
        self,
        jamo_manager=get_jamo_manager,
    ):
        self.model = jamo_manager
        self.block_size = env.BLOCK_SIZE

    @torch.inference_mode()
    def generate_idx(
        self,
        idx: torch.Tensor,
        max_token: int,
        temperature: float=0.8, 
        top_k: int=15,
        eos_id=2
    ) -> T.List[list]:
        T = idx.size(0)
        T_new = T + max_token
        max_seq_length = min(T_new, self.block_size)

        device, dtype = idx.device, idx.dtype
        empty = torch.empty(T_new, dtype=dtype, device=device)
        empty[:T] = idx
        idx = empty
        input_pos = torch.arange(0, T, device=device)

        # generate max_new_tokens tokens
        for _ in range(max_token):
            # forward
            x = idx.index_select(0, input_pos).view(1, -1)

            idx_next = self.model.predict(input=x, max_seq_length=max_seq_length, input_pos=input_pos, temperature=temperature, top_k=top_k)
            
            if idx_next == eos_id:
                break
            
            input_pos = input_pos[-1:] + 1
            idx = idx.index_copy(0, input_pos, idx_next)

        self.model.clean_cache()
        return idx[:input_pos]
    
       # Generate the idx by idx.
    @torch.inference_mode()
    def streaming_generate_idx(
        self,
        idx: torch.Tensor,
        max_token: int,
        temperature: float=0.8, 
        top_k: int=15,
        eos_id=2
    ) -> T.List[list]:
        T = idx.size(0)
        T_new = T + max_token
        max_seq_length = min(T_new, self.block_size)

        device, dtype = idx.device, idx.dtype
        empty = torch.empty(T_new, dtype=dtype, device=device)
        empty[:T] = idx
        idx = empty
        input_pos = torch.arange(0, T, device=device)

        # generate max_new_tokens tokens
        for i in range(max_token):
            x = idx.index_select(0, input_pos).view(1, -1)

            idx_next = self.model.predict(input=x, max_seq_length=max_seq_length, input_pos=input_pos, temperature=temperature, top_k=top_k)

            if idx_next == eos_id:
                break
            
            input_pos = input_pos[-1:] + 1
            idx = idx.index_copy(0, input_pos, idx_next)

            if (i+1)%3==0:
                yield idx[:input_pos]

        self.model.clean_cache()
        yield idx[:input_pos]
        yield None