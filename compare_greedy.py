import os

import torch

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from nanovllm.engine.sequence import Sequence

TAU = 1e-6
MAX_TOKENS = 512

def sanity_check(llm:LLM)->bool:
    logits = torch.tensor([1.,5.,2.],dtype=torch.float).cuda()
    temperatures = torch.tensor(TAU,dtype=torch.float).cuda()
    sample_tokens1 = llm.model_runner.sampler(logits,temperatures)
    sample_tokens2 = llm.model_runner.sampler(logits, temperatures)

    if sample_tokens1 != sample_tokens2:
        print("两次输出不同")
        return False
    elif sample_tokens1.item() != torch.argmax(logits).item():
        print("输出 != logits 的 argmax 位")
        return False
    return True

def install_counters(llm:LLM):
    count = {'decode': 0, 'spec': 0}
    accept_dist = {}
    orig_schedule = llm.scheduler.schedule
    orig_postprocess = llm.scheduler.postprocess

    def wrapped_schedule():
        seqs, is_prefill, is_spec_decode = orig_schedule()
        if not is_prefill:
            count['decode'] += 1
        if is_spec_decode:
            count['spec'] += 1
        return seqs, is_prefill, is_spec_decode

    def wrapped_postprocess(seqs: list[Sequence], token_ids: list[int], is_prefill: bool,
                            token_start_indices: list[int]):
        starts = token_start_indices + [None]
        if not is_prefill:
            for i in range(len(seqs)):
                n = len(token_ids[starts[i]:starts[i + 1]])
                accept_dist[n] = accept_dist.get(n, 0) + 1
        return orig_postprocess(seqs, token_ids, is_prefill, token_start_indices)

    llm.scheduler.schedule = wrapped_schedule
    llm.scheduler.postprocess = wrapped_postprocess

    def restore():
        llm.scheduler.schedule = orig_schedule
        llm.scheduler.postprocess = orig_postprocess

    return count, accept_dist, restore

def run_off(llm:LLM, prompts:list[str],sampling_params:SamplingParams | list[SamplingParams]):
    count_a,_,restore = install_counters(llm)
    orig = llm.model_runner.sample_drafts

    llm.model_runner.sample_drafts = lambda seqs: None
    outputs = llm.generate(prompts, sampling_params,use_tqdm=False)
    llm.model_runner.sample_drafts = orig
    restore()

    return outputs,count_a

def run_on(llm:LLM, prompts:list[str],sampling_params:SamplingParams | list[SamplingParams]):
    count_b, accept_dist, restore = install_counters(llm)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    restore()

    return outputs, count_b, accept_dist

def check_result(llm, tokens_a, tokens_b, count_a, count_b) -> bool:
    """三层：对照成立吗 → 两路相等吗 → （不等时）分歧在协议内吗"""
    # 第一层：对照成立吗（不成立报"无效"——这不是 FAIL，是白比了）
    if count_b["spec"] * 2 <= count_b["decode"]:
        print(f"对照无效：B 路 spec {count_b['spec']} / decode {count_b['decode']}——B 没真投机，换 prompt 重跑")
        return False
    if count_a["spec"] != 0:
        print(f"对照无效：A 路 spec = {count_a['spec']} ≠ 0——off 的 patch 没生效，回查 run_off")
        return False

    # 第二层：主断言
    if tokens_a == tokens_b:
        return True

    # 第三层：主断言红了——先看是"整段前缀等、只是长度差"还是"中间岔了"
    n = min(len(tokens_a), len(tokens_b))
    first_diff = next((i for i in range(n) if tokens_a[i] != tokens_b[i]), None)

    if first_diff is None:                        # 公共前缀全同，只是长度差
        k = llm.model_runner.drafter.k           # 长度差上限，从引擎读真值
        if abs(len(tokens_a) - len(tokens_b)) <= k:
            print(f"前缀全等、长度差 {abs(len(tokens_a) - len(tokens_b))} ≤ k={k}（越界机制，协议内）——降级 PASS")
            return True
        print(f"FAIL: 前缀全等但长度差 {abs(len(tokens_a) - len(tokens_b))} > k={k}——异常越界")
        return False

    lo = max(0, first_diff - 10)                  # 前缀中间岔了：定位 + 取证
    print(f"FAIL: 首分歧位 {first_diff} / 公共长 {n}")
    print(f"  A[{lo}:{first_diff + 10}] = {tokens_a[lo:first_diff + 10]}")
    print(f"  B[{lo}:{first_diff + 10}] = {tokens_b[lo:first_diff + 10]}")
    print(f"  len_a={len(tokens_a)}, len_b={len(tokens_b)}；归因按简报②6：a 真 bug / b 浮点差 / c 温度")
    return False


def report(tokens_a, tokens_b, count_a, count_b, accept_dist, ok):
    """功能块 5：结果汇总打印"""
    stop = lambda t: "eos" if len(t) < MAX_TOKENS else "max"
    spec_pct = count_b["spec"] / count_b["decode"] * 100 if count_b["decode"] else 0.
    print("-" * 60)
    print(f"[A off] {len(tokens_a):4d} tokens | spec {count_a['spec']}/{count_a['decode']} | stop={stop(tokens_a)}")
    print(
        f"[B on ] {len(tokens_b):4d} tokens | spec {count_b['spec']}/{count_b['decode']} ({spec_pct:.0f}%) | stop={stop(tokens_b)}")
    print(f"accept dist (m+1 -> rounds): {dict(sorted(accept_dist.items()))}")
    print("-" * 60)
    if ok:
        print(f"PASS: token-by-token identical ({len(tokens_a)} tokens)")
    else:
        print("RESULT: FAIL / 对照无效 —— 终态与分歧细节见上方输出")

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path,enforce_eager=True)

    if not sanity_check(llm):
        print("sanity自检报错")
        return

    sampling_params = SamplingParams(TAU, MAX_TOKENS)
    prompts = [
        "把下面这句话原样重复 10 遍：你好呀，你好帅",
        # "从 1 依次数到 60，每个数字单独一行",
    ]
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                      tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
        for prompt in prompts
    ]

    outputs_off, count_a = run_off(llm,prompts,sampling_params)
    outputs_on, count_b, accept_dist = run_on(llm,prompts,sampling_params)
    tokens_a = outputs_off[0]["token_ids"]
    tokens_b = outputs_on[0]["token_ids"]

    ok = check_result(llm, tokens_a, tokens_b, count_a, count_b)

    report(tokens_a, tokens_b, count_a, count_b, accept_dist, ok)

if __name__ == '__main__':
    main()

