from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def allocate_block(self,token_ids:list[int],h:int)-> tuple[Block,bool]:
        h = self.compute_hash(token_ids,h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h,-1)
        cache_miss = False
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = self._allocate_block()
            block = self.blocks[block_id]
        else:
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self.blocks[block_id]
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
        if h != -1:
            block.update(h,token_ids)
            self.hash_to_block_id[h] = block_id
        return block,cache_miss

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate_prefill(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate_prefill(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_prefill_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_prefill_tokens = 0
        seq.block_table.clear()

    def can_allocate_decode(self,seq:Sequence):
        return len(self.free_block_ids) >= (seq.num_blocks - len(seq.block_table))

    def allocate_decode(self,seq:Sequence):
        block_table = seq.block_table
        last_block:Block = self.blocks[block_table[-1]]
        num_cached_blocks = len(block_table)
        num_total_blocks = seq.num_blocks  # num_blocks 是 @property（sequence.py:76-78）——property 的意义就是"看起来像属性、用起来像属性”：访问不加括号

        token_ids = seq.block(num_cached_blocks - 1)
        if last_block.hash == -1 and len(token_ids) == self.block_size:
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids,prefix)
            last_block.update(h,token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        h = last_block.hash
        for i in range(num_cached_blocks,num_total_blocks):
            token_ids = seq.block(i)
            block,_ = self.allocate_block(token_ids, h)
            assert isinstance(block,Block)
            seq.block_table.append(block.block_id)
            h = block.hash

    def deallocate_decode(self,seq:Sequence):
        block_table = seq.block_table
        num_reject_blocks = len(block_table) - seq.num_blocks

        if num_reject_blocks > 0:
            for i in range(1,num_reject_blocks + 1):
                block = self.blocks[block_table[-i]]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(block.block_id)
            del block_table[-num_reject_blocks:]

        if seq.last_block_num_tokens > 0 and seq.last_block_num_tokens < seq.block_size:
            last_block = self.blocks[block_table[-1]]
            last_block.hash = -1
            last_block.token_ids = seq.block(seq.num_blocks-1)

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_prefill_tokens // self.block_size
        end = (seq.num_cached_prefill_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
