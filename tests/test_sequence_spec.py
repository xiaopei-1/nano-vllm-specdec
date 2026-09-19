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