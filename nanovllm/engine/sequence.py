from copy import copy
from enum import Enum, auto
from itertools import count

import torch

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_prefill_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        self.spec_tokens = []
        self.num_spec_tokens = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    def extend_tokens(self,token_ids:list[int])->None:
        num_unaccept_draft_tokens = self.num_spec_tokens - len(token_ids) + 1
        if num_unaccept_draft_tokens > 0:
            del self.token_ids[-num_unaccept_draft_tokens:]
        self.num_tokens -= num_unaccept_draft_tokens
        self.append_token(token_ids[-1])
        self.num_spec_tokens = 0

    def merge_spec_tokens(self)->bool:
        if len(self.spec_tokens) == 0:
            return False
        self.num_spec_tokens = len(self.spec_tokens)
        self.token_ids.extend(self.spec_tokens)
        self.num_tokens += self.num_spec_tokens
        self.last_token = self.token_ids[-1]
        self.spec_tokens = []
        return True

    def unmerge_spec_tokens(self):
        if self.num_spec_tokens <= 0:
            return
        del self.token_ids[-self.num_spec_tokens:]
        self.num_tokens -= self.num_spec_tokens
        self.last_token = self.token_ids[-1]
        self.spec_tokens = []
        self.num_spec_tokens = 0

    # Properties for speculative decode

    @property
    def num_decode_tokens(self):
        return self.num_spec_tokens + 1

    @property
    def decode_tokens(self):
        return self.token_ids[-self.num_spec_tokens - 1:]

    @property
    def decode_draft_tokens(self) -> list[int]:
        return self.token_ids[-self.num_spec_tokens :] if self.num_spec_tokens > 0 else []

    @property
    def decode_position(self):
        return torch.arange(self.num_tokens - self.num_spec_tokens - 1, self.num_tokens)

    @property
    def decode_blocks(self) -> list:
        num_decode_blocks = 1 + (self.num_spec_tokens - self.last_block_num_tokens + self.block_size) // self.block_size
        return self.block_table[- num_decode_blocks:]

    @property
    def decode_slot_mapping(self):
        # decode blocks
        decode_blocks = torch.as_tensor(self.decode_blocks,dtype=torch.int32)
        block_size = self.block_size
        num_blocks = decode_blocks.numel()

        # block token counts
        counts = torch.full((num_blocks,), block_size, dtype = torch.int32)
        counts[-1] = self.last_block_num_tokens

        # repeat block offsets
        block_offsets = torch.repeat_interleave(decode_blocks * block_size,repeats=counts)

        # in-block offsets via cumsum
        counts_cum = torch.cumsum(counts,dim=0)
        starts = torch.repeat_interleave(counts_cum - counts,counts)
        in_block_offsets = torch.arange(counts.sum(), dtype = torch.int32) - starts

        slot_mapping = block_offsets + in_block_offsets
        return slot_mapping[- self.num_spec_tokens - 1:]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_prefill_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_prefill_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
