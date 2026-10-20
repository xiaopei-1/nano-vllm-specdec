from nanovllm.engine.sequence import Sequence

eos_id = -1

def test_case1_accept_m2():
    # 用例1：m=2，bonus=80 → 留前 m+1
    seq = Sequence([10,20,30,40])
    seq.spec_tokens = [50,60,70]
    seq.merge_spec_tokens()
    seq.extend_tokens([50,60,80])
    assert seq.token_ids == [10, 20, 30, 40, 50, 60, 80]
    assert seq.num_tokens == 7
    assert seq.num_spec_tokens == 0  # 用例4 附加断言
    assert seq.num_tokens == len(seq.token_ids)  # 用例4 附加断言

def test_case2():
    # 用例2：m=0 全拒，bonus=80
    seq = Sequence([10,20,30,40])
    seq.spec_tokens = [50,60,70]
    seq.merge_spec_tokens()
    seq.extend_tokens([80])
    assert seq.token_ids == [10, 20, 30, 40, 80]
    assert seq.num_tokens == 5
    assert seq.num_spec_tokens == 0  # 用例4 附加断言
    assert seq.num_tokens == len(seq.token_ids)  # 用例4 附加断言

def test_case3():
    # 用例3：bonus 是 eos_id
    seq = Sequence([10,20,30,40])
    seq.spec_tokens = [50,60,70]
    seq.merge_spec_tokens()
    seq.extend_tokens([50,60,eos_id])
    assert seq.token_ids == [10, 20, 30, 40, 50, 60, eos_id]
    assert seq.num_tokens == 7
    assert seq.num_spec_tokens == 0  # 用例4 附加断言
    assert seq.num_tokens == len(seq.token_ids)  # 用例4 附加断言

def test_case5():
    # 用例5：m=k 用例
    seq = Sequence([10,20,30,40])
    seq.spec_tokens = [50,60,70,80]
    seq.merge_spec_tokens()
    seq.extend_tokens([50,60,70,80,90])
    assert seq.token_ids == [10, 20, 30, 40, 50, 60, 70,80,90]
    assert seq.num_tokens == 9
    assert seq.num_spec_tokens == 0  # 用例4 附加断言
    assert seq.num_tokens == len(seq.token_ids)  # 用例4 附加断言

def test_case6_unmerge_revert():
    # 用例6（R1 回归锁）：merge → unmerge 完全回滚——unmerge 是 merge 的严格逆操作
    # preempt 场景：已 merge 未 verify 的序列被抢占，草稿必须撤干净，否则重 prefill 时
    # 草稿被当正文（#147 缺陷 5）。四字段全部回到 merge 前（AI 代写经授权 2026-10-20）
    seq = Sequence([10,20,30,40])
    seq.spec_tokens = [50,60,70]
    seq.merge_spec_tokens()
    assert seq.num_tokens == 7 and seq.last_token == 70   # merge 确实生效了再谈回滚
    seq.unmerge_spec_tokens()
    assert seq.token_ids == [10, 20, 30, 40]
    assert seq.num_tokens == 4
    assert seq.num_tokens == len(seq.token_ids)   # num_tokens 是独立字段——同步减是 unmerge 的头号坑位
    assert seq.last_token == 40                    # last_token 恢复到锚 token
    assert seq.num_spec_tokens == 0

def test_case7_unmerge_idempotent():
    # 用例7（R1 回归锁）：未 merge（num_spec_tokens=0）时 unmerge 无副作用
    # preempt 也可能打到没有草稿的序列——幂等保护不能让状态变坏
    seq = Sequence([10,20,30,40])
    seq.unmerge_spec_tokens()
    assert seq.token_ids == [10, 20, 30, 40]
    assert seq.num_tokens == 4
    assert seq.last_token == 40
    assert seq.num_spec_tokens == 0