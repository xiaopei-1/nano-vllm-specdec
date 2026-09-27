from nanovllm.spec_decode.ngram_proposer import NgramProposer

def test_case1():
    # 历史有重复
    ngram_proposer = NgramProposer()
    tokens = [1,2,3,4,5,6,2,3,4,7,8,3,4,9,10,2,3,4]
    proposed_tokens = ngram_proposer.propose(tokens,2,2,20,4)
    assert proposed_tokens == [9,10]

def test_case2():
    # 无匹配
    ngram_proposer = NgramProposer()
    tokens = [1,2,3,4,5,6]
    proposed_tokens = ngram_proposer.propose(tokens,2,2,20,4)
    assert proposed_tokens == []

def test_case3():
    # 输入太短
    ngram_proposer = NgramProposer()
    tokens = [1]
    proposed_tokens = ngram_proposer.propose(tokens,2,2,20,4)
    assert proposed_tokens == []

def test_case4():
    # k被max_model_len部分截断：差值 11-10=1，草稿长度=差值
    ngram_proposer = NgramProposer()
    tokens = [1,2,3,4,5,6,3,4,5,6]
    proposed_tokens = ngram_proposer.propose(tokens,2,2,11,4)
    assert proposed_tokens == [3]

def test_case5():
    # 候选段不足k
    ngram_proposer = NgramProposer()
    tokens = [1,2,3,2,3]
    proposed_tokens = ngram_proposer.propose(tokens,2,2,20,4)
    assert proposed_tokens == [2,3]

def test_case6():
    # 全默认参数：锁 max_model_len 回退（对 #147 的差异改动，防照抄原文回退丢失）
    ngram_proposer = NgramProposer()
    tokens = [1,2,3,4,5,6,2,3,4,7,8,3,4,9,10,2,3,4]
    proposed_tokens = ngram_proposer.propose(tokens)
    assert proposed_tokens == [7,8,3,4]