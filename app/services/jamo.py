import typing as T
from pathlib import Path
import sys
import torch

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from managers import (
    JamoModelManager
)

class JamoService:
    def __init__(
        self,
    ):
        self.model = JamoModelManager()
        self.block_size = 256

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
        
        if type(max_token) != type(1) or max_token <= 0:
            raise ValueError("Please, check you sent max_token query.")

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
        top_k: int=20,
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
            print(i)
            x = idx.index_select(0, input_pos).view(1, -1)

            idx_next = self.model.predict(input=x, max_seq_length=max_seq_length, input_pos=input_pos, temperature=temperature, top_k=top_k)

            if idx_next == eos_id:
                break
            
            input_pos = input_pos[-1:] + 1
            idx = idx.index_copy(0, input_pos, idx_next)

            if (i+1)%3==0:
                yield idx[:input_pos]

        print("clean cache")
        self.model.clean_cache()
        print("return tokens lastly")
        yield idx[:input_pos]
        print("for eos")
        yield None