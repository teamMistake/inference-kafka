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
        model_path: str="model_store/jamo.tar"
    ):
        self.model = JamoModelManager(model_path=model_path)
        self.block_size = 256

    @torch.inference_mode()
    def multibatch_generate(
        self,
        idx: torch.Tensor,
        max_token: int,
        temperature: float=0.8, 
        top_k: int=15,
        eos_id=2
    ) -> T.List[T.List[list]]:
        self.safety(max_token)

        B = idx.size(0)
        T = idx.size(1) # [batch, T, seq]
        T_new = T + max_token
        max_seq_length = min(T_new, self.block_size)

        device, dtype = idx.device, idx.dtype
        empty = torch.empty((B, T_new), dtype=dtype, device=device)
        empty[range(B), :T] = idx
        idx = empty # [B, T, 256 or T_new]
        input_pos = torch.arange(0, T, device=device)
        # input_pos = torch.stack([pos, pos, pos], dim=0)

        finished_idxs = torch.zeros(B)
        
        # generate max_new_tokens tokens
        for i in range(max_token):
            # forward
            x = idx.index_select(1, input_pos).view(B, -1)
            idx_next = self.model.predict(input=x, max_seq_length=max_seq_length, input_pos=input_pos, temperature=temperature, top_k=top_k, multibatch=True)

            input_pos = input_pos[-1:] + 1
            idx = idx.index_copy(1, input_pos, idx_next)

            idx_next = idx_next.squeeze(1)
            # if all tensor look like [1, 1, 1] and break the loop
            finished_idxs[idx_next==eos_id] = input_pos
            if not torch.any(finished_idxs==0):
                break

        self.model.clean_cache()
        return idx[:, :input_pos], finished_idxs.tolist()

    @torch.inference_mode()
    def generate_idx(
        self,
        idx: torch.Tensor,
        max_token: int,
        temperature: float=0.8, 
        top_k: int=15,
        eos_id=2
    ) -> T.List[list]:        
        self.safety(max_token)

        T = idx.size(0)
        T_new = T + max_token
        max_seq_length = min(T_new, self.block_size) - 2

        device, dtype = idx.device, idx.dtype
        empty = torch.empty(T_new, dtype=dtype, device=device)
        empty[:T] = idx
        idx = empty
        input_pos = torch.arange(0, T, device=device)

        # generate max_new_tokens tokens
        for _ in range(max_seq_length):
            # forward
            x = idx.index_select(0, input_pos).view(1, -1)
            idx_next = self.model.predict(input=x, max_seq_length=max_seq_length, input_pos=input_pos, temperature=temperature, top_k=top_k)

            if idx_next == eos_id:
                break
            
            input_pos = input_pos[-1:] + 1
            idx = idx.index_copy(0, input_pos, idx_next)

        self.model.clean_cache()
        return idx[:input_pos]

    @torch.inference_mode()
    def multibatch_streaming(
        self,
        idx: torch.Tensor,
        max_token: int,
        temperature: float=0.8, 
        top_k: int=15,
        eos_id=2
    ) -> T.List[T.List[list]]:
        self.safety(max_token)

        B = idx.size(0)
        T = idx.size(1) # [batch, T, seq]
        T_new = T + max_token
        max_seq_length = min(T_new, self.block_size)

        device, dtype = idx.device, idx.dtype
        empty = torch.empty((B, T_new), dtype=dtype, device=device)
        empty[range(B), :T] = idx
        idx = empty # [B, T, 256 or T_new]
        input_pos = torch.arange(0, T, device=device)
        # input_pos = torch.stack([pos, pos, pos], dim=0)

        finished_idxs = torch.zeros(B)
        
        # generate max_new_tokens tokens
        for i in range(max_seq_length):
            # forward
            x = idx.index_select(1, input_pos).view(B, -1)
            idx_next = self.model.predict(input=x, max_seq_length=max_seq_length, input_pos=input_pos, temperature=temperature, top_k=top_k, multibatch=True)

            input_pos = input_pos[-1:] + 1
            idx = idx.index_copy(1, input_pos, idx_next)

            idx_next = idx_next.squeeze(1)
            # if all tensor look like [1, 1, 1] and break the loop
            finished_idxs[idx_next==eos_id] = input_pos
            if not torch.any(finished_idxs==0):
                break

            if (i+1) % 3 == 0:
                yield idx[:, :input_pos], finished_idxs.tolist()

        self.model.clean_cache()
        yield idx[:, :input_pos], finished_idxs.tolist()
        yield None, None
    
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
        self.safety(max_token)

        T = idx.size(0)
        T_new = T + max_token
        max_seq_length = min(T_new, self.block_size)

        device, dtype = idx.device, idx.dtype
        empty = torch.empty(T_new, dtype=dtype, device=device)
        empty[:T] = idx
        idx = empty
        input_pos = torch.arange(0, T, device=device)

        # generate max_new_tokens tokens
        for i in range(max_seq_length):
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

    def safety(self, max_token):
        if type(max_token) != type(1) or max_token <= 0:
            raise ValueError("Please, check you sent max_token query.")
        