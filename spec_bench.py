import os
import sys
import time

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence

TAU = 1e-6
MAX_TOKENS = 512

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

def build_llm(util: float =0.9) -> LLM:
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    return LLM(path, enforce_eager=True, gpu_memory_utilization=util)

def chat_wrap(llm: LLM,prompt: str) -> str:
    return llm.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                      tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)

def run_bench():
    buckets = [
        ("复述", "把下面这句话原样重复 10 遍：你好呀，你好帅"),
        ("数数", "从 1 依次数到 60，每个数字单独一行"),
        ("开放", "写一段 200 字左右的内容介绍你自己"),
    ]
    llm = build_llm()
    sp = SamplingParams(temperature=TAU, max_tokens=512)
    for name, raw in buckets:
        prompt = [chat_wrap(llm,raw)]
        t0 = time.perf_counter(); out_a, _ = run_off(llm,prompt,sp)
        ta = time.perf_counter() - t0
        llm.scheduler.block_manager.hash_to_block_id.clear()

        t0 = time.perf_counter(); out_b, cnt, dist = run_on(llm, prompt, sp)
        tb = time.perf_counter() - t0
        na, nb = len(out_a[0]["token_ids"]), len(out_b[0]["token_ids"])
        print(f"[{name}] A: {na} tok / {ta:.1f}s = {na / ta:.1f} tok/s | "
              f"B: {nb} tok / {tb:.1f}s = {nb / tb:.1f} tok/s | x{nb / tb / (na / ta):.2f} | "
              f"spec {cnt['spec']}/{cnt['decode']} | dist {dict(sorted(dist.items()))}")

def run_r1():
    llm = build_llm(util=0.35)
    sp = SamplingParams(temperature=TAU, max_tokens=512)
    prompts = [chat_wrap(llm, "把下面这句话原样重复 10 遍：你好呀，你好帅")] * 8
    preempt_log = []
    orig_preempt = llm.scheduler.preempt
    def wrapped_preempt(seq):
        preempt_log.append(seq.num_spec_tokens)
        return orig_preempt(seq)
    llm.scheduler.preempt = wrapped_preempt
    out_small, cnt, _ = run_on(llm, prompts, sp)
    llm.scheduler.preempt = orig_preempt
    print(f"preempt 次数 {len(preempt_log)}，各次 num_spec_tokens: {preempt_log}")
    print(f"  其中带草稿被抢（>0）的次数：{sum(1 for x in preempt_log if x > 0)}")
    del llm; torch.cuda.empty_cache()
    llm2 = build_llm()
    preempt2 = []
    orig2 = llm2.scheduler.preempt
    def wrapped2(seq):
        preempt2.append(seq.num_spec_tokens)
        return orig2(seq)
    llm2.scheduler.preempt = wrapped2
    out_true, _, _ = run_on(llm2, prompts, sp)
    llm2.scheduler.preempt = orig2
    print(f"大池 preempt 次数 {len(preempt2)}（应为 0——真值前提)")
    ok = all(a["token_ids"] == b["token_ids"] for a, b in zip(out_small, out_true))
    print("小池 vs 大池逐 token：" + ("全等（绿）" if ok else "存在不等(对照9简报2-6归因)"))

def run_hash():
    llm = build_llm(util=0.35)
    sp = SamplingParams(temperature=TAU, max_tokens=512)
    prompts = [chat_wrap(llm, "把下面这句话原样重复 10 遍：你好呀，你好帅")] * 8
    stats = {"calls": 0, "cached": 0, "demand": 0}
    orig_probe = llm.scheduler.block_manager.can_allocate_prefill
    def wrapped_probe(seq):
        r = orig_probe(seq)
        stats["calls"] += 1
        stats["cached"] += max(0, r)
        stats["demand"] += seq.num_blocks
        return r
    llm.scheduler.block_manager.can_allocate_prefill = wrapped_probe
    run_on(llm, prompts, sp)
    llm.scheduler.block_manager.can_allocate_prefill = orig_probe
    print(f"prefill 探测 {stats['calls']} 次，需求 {stats['demand']} 块，命中 {stats['cached']} 块，"
          f"命中率 {stats['cached']/stats['demand']*100:.1f}%")

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'bench'
    modes = {
        'bench': run_bench,
        'r1': run_r1,
        'hash': run_hash,
    }
    if mode not in modes:
        print(f"未知模式 {mode}，可用：{'/'.join(modes)}")
        return
    modes[mode]()

if __name__ == '__main__':
    main()