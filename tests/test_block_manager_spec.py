from nanovllm.engine.block_manager import BlockManager, Block
from nanovllm.engine.sequence import Sequence

# 小世界：块大小 4、共 8 块。prompt token id 从 100 起、草稿 id 从 200 起（与块 id 空间分开）
Sequence.block_size = 4


def new_seq(num_prompt: int) -> tuple[BlockManager, Sequence]:
    """建一个 prefill 完成的序列：探测缓存 → 执行分配 → block_table 就位"""
    bm = BlockManager(8, 4)
    seq = Sequence(list(range(100, 100 + num_prompt)))
    num_cached = bm.can_allocate_prefill(seq)
    bm.allocate_prefill(seq, num_cached)
    return bm, seq


def merge_with(seq: Sequence, num_spec: int):
    """给序列挂 num_spec 个草稿并 merge（token_ids 变长、num_blocks 涨、block_table 不动）"""
    seq.spec_tokens = list(range(200, 200 + num_spec))
    seq.merge_spec_tokens()


def run_verify_round(bm: BlockManager, seq: Sequence, num_spec: int, accepted: list[int]):
    """完整 spec 轮次：merge → 预分配 → verify 结算（extend 删拒）→ 回滚。
    accepted = 最终保留的 token（m+1 个：m 个接受的草稿 + 1 个纠错/bonus）"""
    merge_with(seq, num_spec)
    bm.allocate_decode(seq)
    seq.extend_tokens(accepted)
    bm.deallocate_decode(seq)


def test_case1():
    # 预分配差额：6 prompt（2 块）+ 6 草稿 → merge 后 12 token → num_blocks=3，差额 1
    bm, seq = new_seq(6)
    merge_with(seq, 6)
    # TODO 断言：can_allocate_decode 此时返回什么？（free = 8-2 = 6，差额 = 3-2 = 1）
    assert bm.can_allocate_decode(seq) == True
    bm.allocate_decode(seq)
    # TODO 断言1（I3 长度同步）：block_table 长度现在是几？等于 seq.num_blocks 吗？
    assert len(seq.block_table) == 3
    # TODO 断言2（free 计数）：free 还剩几个块？
    assert len(bm.free_block_ids) == 5

def test_case2():
    # 草稿未跨界：6 prompt（2 块）+ 1 草稿 → merge 后 7 token → num_blocks 仍 2，差额 0
    bm, seq = new_seq(6)
    merge_with(seq, 1)
    # TODO 断言：can_allocate_decode 返回什么？
    assert bm.can_allocate_decode(seq) == True
    bm.allocate_decode(seq)
    # TODO 断言：block_table 长度变了吗？free 少了吗？（注意末块 3/4 未满——fill 段不会登记）
    assert len(seq.block_table) == 2
    assert len(bm.free_block_ids) == 6

def test_case3():
    # 回滚 + 末块恰好满：6 prompt + 6 草稿（预分配后 3 块），verify 只接受 1 个草稿（m=1）
    # extend 后 num_tokens = 12-5+1 = 8 → num_blocks = 2 → 差额块还回去
    bm, seq = new_seq(6)
    merge_with(seq, 6)
    bm.allocate_decode(seq)
    seq.extend_tokens([200, 999])          # 接受草稿 200 + 纠错 999（m=1 协议：参数 m+1 个）
    bm.deallocate_decode(seq)
    # TODO 断言1（I3 恢复）：block_table 长度 == seq.num_blocks == 几？
    assert len(seq.block_table) == seq.num_blocks == 2
    # TODO 断言2（free 恢复）：free 回到预分配前的几个？
    assert len(bm.free_block_ids) == 6
    # TODO 断言3（末块恰满 → hash 保留，与 #147 的差异点正面锁）：
    #   回滚后末块是 token[4:8] = [104,105,200,999]，恰好 4/4 满。
    #   想清楚再断：bm.blocks[seq.block_table[-1]].hash 应该是 -1 还是保留？
    #   提示：verify 前向写进这块 KV 的是 merge 时刻内容 [104,105,200,201] 的——
    #   hash/token_ids/KV 三者冻结在一致状态，缓存依然有效
    assert bm.blocks[seq.block_table[-1]].hash != -1

def test_case4():
    # 末块未满 → hash 作废：5 prompt（2 块）+ 4 草稿 → merge 后 9 token → 3 块
    # 注意 merge 把末块 [104] 填成 [104,200,201,202] 恰好满——allocate_decode 会登记它
    # 拒到 m=1 后 num_tokens = 9-3+1 = 7 → 2 块，末块 [104,200,999] 只有 3/4——hash 必须作废
    bm, seq = new_seq(5)
    run_verify_round(bm, seq, 4, [200, 999])
    # 断言1：末块未满 → hash 作废
    assert bm.blocks[seq.block_table[-1]].hash == -1
    # 断言2（I3/free）：2 块、free 恢复到 6
    assert len(seq.block_table) == seq.num_blocks == 2
    assert len(bm.free_block_ids) == 6
    # 断言3（B 档已清）：末块重置时映射同步删除，字典不残留 sta
    assert len(bm.hash_to_block_id) == 0

def test_case5():
    # stale hash 回归锁：重演场景 4 的末状态，然后用同样内容再查一次缓存
    bm, seq = new_seq(5)
    run_verify_round(bm, seq, 4, [200, 999])
    # B 档后字典里已无 stale 映射（场景 4 重置时已删）——查询直接 miss；
    # 断言 miss == True 仍锁住"不误命中"，只是机制从"比中兜底"变为"无可映射命中"
    block, miss = bm.allocate_block([104, 200, 201, 202], -1)
    # 断言：同 hash 命中 stale 映射 → token_ids 比对兜底 → 不误命中（A 档核心行为）
    assert miss == True
    # 加强：miss 路径拿到的新块被 update 成查询内容（缓存从此指向新块，stale 被覆盖）
    assert block.token_ids == [104, 200, 201, 202]
