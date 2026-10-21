import torch
from nanovllm.layers.rejection_sampler import RejectionSampler

VOCAB = 100

# T=0 greedy 替身：直接取 argmax。
# 为什么不用真 Sampler：它带 @torch.compile，本地 Windows 无 MSVC 编译器会
# InductorError 硬炸；M0 测的是 rejection 比较逻辑而非采样逻辑——
# RejectionSampler 构造收 sampler 是依赖注入，正好允许换替身（真 Sampler 路径 M2 云端验）
greedy = lambda logits, temperatures: logits.argmax(dim=-1)

# M4-B 固定值替身：恒返 88（与 make_logits 构造的 argmax 位错开）。
# 用途：区分"走了 sampler 路径"还是"被 greedy argmax 覆盖"——
# greedy 替身的返回恰等于 argmax，两条分流路径输出相同，测不出分流；88 才能分辨
stub = lambda logits, temperatures: torch.full((logits.shape[0],), 88, dtype=torch.int64)


def make_logits(argmax_ids: list[int]) -> torch.Tensor:
    """每行目标位 +10、其余 -10，使每行 argmax 恰为给定 id"""
    logits = torch.full((len(argmax_ids), VOCAB), -10.0)
    for r, t in enumerate(argmax_ids):
        logits[r, t] = 10.0
    return logits


def run(argmax_ids, draft_tokens, cum_draft, cum_sampled, temperatures=None, sampler=greedy):
    """构造 CPU 张量直调 rejection_sample（绕开 prepare_tensors 的 .cuda()）
    返回 (output_tokens, num_accepted_tokens)
    temperatures：None=全 0 占位（float32）。传入时必须是 float 张量——
      全 int 列表会造出 int64，引擎 masked_fill_ 填 1.0 直接炸（真实链路恒 float32）
    sampler：默认 greedy 替身（既有用例行为不变）；测分流时换 stub"""
    rs = RejectionSampler(sampler)
    logits = make_logits(argmax_ids)
    temperatures = torch.zeros(len(argmax_ids)) if temperatures is None else temperatures  # M4-B 起分流消费它
    draft = torch.tensor(draft_tokens, dtype=torch.int64)
    cd = torch.tensor(cum_draft, dtype=torch.int64)
    cs = torch.tensor(cum_sampled, dtype=torch.int64)
    batch = len(cum_draft) - 1  # 首 0 哨兵让表长 = 序列数 + 1
    max_sampled = max(cum_sampled[i + 1] - cum_sampled[i] for i in range(batch))
    output = torch.zeros((batch, max_sampled), dtype=torch.int64)
    num_accepted = torch.zeros((batch,), dtype=torch.int64)
    return rs.rejection_sample(logits, temperatures, draft, cd, cs, output, num_accepted)


def test_case1():
    # 全接受：单序列 4 draft 全对
    # sampled 布局 = [验d0, 验d1, 验d2, 验d3, bonus]，argmax 依次设为 [5,6,7,8,9]
    output, num_accepted = run(
        argmax_ids=[5, 6, 7, 8, 9],
        draft_tokens=[5, 6, 7, 8],
        cum_draft=[0, 4],
        cum_sampled=[0, 5],
    )
    # TODO 断言1：output 第 0 行的 5 个元素是什么？
    assert output[0].tolist() == [5,6,7,8,9]
    # TODO 断言2：num_accepted 是什么？
    assert num_accepted.tolist() == [5]

def test_case2():
    # 部分接受：第 2 位断。argmax[2]=4 而 draft[2]=7 → j=2 处比较失败
    output, num_accepted = run(
        argmax_ids=[5, 6, 4, 8, 9],
        draft_tokens=[5, 6, 7, 8],
        cum_draft=[0, 4],
        cum_sampled=[0, 5],
    )
    # TODO 断言1：output 第 0 行完整的 5 个元素——注意断点位置写的是谁、其后是零填充
    assert output[0].tolist() == [5,6,4,0,0]
    # TODO 断言2：num_accepted 是什么？
    assert num_accepted.tolist() == [3]

def test_case3():
    # m=0 全拒：第 0 位即断。argmax[0]=3 而 draft[0]=5
    output, num_accepted = run(
        argmax_ids=[3, 6, 7],
        draft_tokens=[5, 6],
        cum_draft=[0, 2],
        cum_sampled=[0, 3],
    )
    # TODO 断言1：m+1 协议下最少产出几个？产出的是什么值？
    assert output[0].tolist() == [3,0,0]
    # TODO 断言2：num_accepted 是什么？
    assert num_accepted.tolist() == [1]

def test_case4():
    # 多序列混合（#147 错位缺陷回归锁）。两条序列，draft 数 [2, 1]：
    #   seq0 draft=[1,2]，seq1 draft=[7]
    # 全批采样布局（5 行）：
    #   行0=验seq0.d0  行1=验seq0.d1  行2=seq0.bonus
    #   行3=验seq1.d0  行4=seq1.bonus
    output, num_accepted = run(
        argmax_ids=[1, 2, 50, 7, 60],
        draft_tokens=[1, 2, 7],
        cum_draft=[0, 2, 3],
        cum_sampled=[0, 3, 5],
    )
    # 两条序列各自"全接受"（正确语义下每个验证位都与 draft 相同）
    # 照 #147 原样实现（比较处用 draft 基准取 sampled）此例应红——
    # 先想清楚：seq1 的 j=0 比较会读到 5 行采样中的第几行？
    # TODO 断言1：output 两行各是什么？
    assert output[0].tolist() == [1,2,50]
    assert output[1].tolist() == [7,60,0]
    # TODO 断言2：num_accepted 两个值是什么？
    assert num_accepted.tolist() == [3,2]

# ===== M4-B greedy 分流用例（2026-10-21，断言值协议手推）=====
# 协议：分流后 sample_tokens = where(T==0 的位取 argmax, T>0 的位取 sampler 结果)

def test_m4_case_a_all_greedy():
    # 全 greedy：温度全 0（默认占位即所需）+ stub 恒返 88。
    # 手推：3 个采样位全被 argmax [5,6,50] 覆盖（88 无效）→
    #   draft [5,6] 前两位接受、第 3 位 bonus → output=[5,6,50], accepted=3
    # 有牙齿：注释引擎分流（sampled 直连）后 sampled 全 88 → j=0 即断 → output=[88,0,0] → 此断言红
    output, num_accepted = run(
        argmax_ids=[5, 6, 50],
        draft_tokens=[5, 6],
        cum_draft=[0, 2],
        cum_sampled=[0, 3],
        sampler=stub,
    )
    assert output[0].tolist() == [5, 6, 50]
    assert num_accepted.tolist() == [3]

def test_m4_case_b_all_sampled():
    # 全采样：温度全 0.7 + stub 恒返 88，argmax 位 [5,6,50] 不生效。
    # 手推：sampled 全 88 → draft [88,88] 前两位接受、第 3 位 bonus=88 →
    #   output=[88,88,88], accepted=3
    # 有牙齿：分流条件若写反（mask 取反），output 会变 [5,6,50] → 此断言红
    output, num_accepted = run(
        argmax_ids=[5, 6, 50],
        draft_tokens=[88, 88],
        cum_draft=[0, 2],
        cum_sampled=[0, 3],
        temperatures=torch.tensor([0.7, 0.7, 0.7]),
        sampler=stub,
    )
    assert output[0].tolist() == [88, 88, 88]
    assert num_accepted.tolist() == [3]

def test_m4_case_c_mixed_batch():
    # 混合批：温度按序列展开，单序列做不到混合——必须两条序列。
    #   seq0：T=0、draft [5,6]、argmax 位 [5,6,50]（mask 真 → argmax 生效）
    #   seq1：T=0.7、draft [7]（mask 假 → 采样路径，stub 恒 88）
    #   采样位布局（5 位）：[0,0,0, 0.7,0.7] = seq0 三位 + seq1 两位
    # 手推：合并后 sample_tokens = [5,6,50, 88,88] →
    #   seq0 全接受 [5,6,50], accepted=3；seq1 首位 7 vs 88 即断 → [88,0,0], accepted=1
    output, num_accepted = run(
        argmax_ids=[5, 6, 50, 70, 71],   # 后两位无意义（mask 假），随便给
        draft_tokens=[5, 6, 7],
        cum_draft=[0, 2, 3],
        cum_sampled=[0, 3, 5],
        temperatures=torch.tensor([0.0, 0.0, 0.0, 0.7, 0.7]),
        sampler=stub,
    )
    assert output[0].tolist() == [5, 6, 50]
    assert output[1].tolist() == [88, 0, 0]
    assert num_accepted.tolist() == [3, 1]
