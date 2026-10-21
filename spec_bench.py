import os
import sys
import time

import atexit
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager

TAU = 1e-6
MAX_TOKENS = 2048
R1_TEXTS = (
    "选择昇腾，本质是押注国产算力的结构性机会，但必须清醒管理两类风险：一是生态不成熟带来的开发摩擦，二是过度绑定单一栈导致的技能锁定。对计算机研究生而言，可取策略是“以昇腾为入口，但不被昇腾定义”：借国产替代缺口入行，同时刻意保持异构计算、编译优化和底层系统能力的可迁移性，把生态不完善期当作积累“填坑经验”的窗口，而非职业终点。这样既能吃到国产化红利，又保留未来转向通用GPU生态的议价能力。",
    "市交通委今日宣布，主城区120个路口已上线AI信号灯优化系统。系统根据实时车流、公交优先和行人过街需求动态调整配时，试运行一个月后，早晚高峰平均拥堵指数下降8.6%，主干道通行速度提升约11%。相关负责人表示，下一步将把医院、学校周边纳入重点场景，并保留人工调度兜底。专家提醒，算法优化不能只看机动车效率，还需兼顾非机动车与行人安全，避免“车快人难行”。市民可通过官方平台反馈路口问题，数据将用于模型迭代。",
    "蜜蜂的舞蹈语言是动物行为学中的经典课文。侦察蜂发现蜜源后，会回到蜂巢跳“8字舞”或“圆圈舞”。圆圈舞通常表示蜜源较近，8字舞则能传递方向和距离：舞蹈中轴线与重力方向的夹角对应太阳方位，摆尾持续时间大致反映距离。科学家发现，不同蜂种舞蹈细节有差异，且蜜源质量也会影响舞蹈强度。蜜蜂并不真正“说话”，而是通过动作和振动把信息编码给同伴。这一现象说明，社会性昆虫也能依靠简单规则完成复杂协作。",
    "在计算机系统中，缓存利用的是程序访问的局部性原理。时间局部性指刚被访问的数据很可能再次被访问，空间局部性指相邻地址的数据常被连续使用。CPU读取数据时先查L1、L2、L3缓存，未命中才访问内存，因此循环顺序、数组布局和数据结构对齐都会影响性能。优化时应尽量顺序访问内存，减少随机跳转，并控制工作集大小以适配缓存容量。但缓存并非越大越好，还需权衡命中率、延迟、功耗和成本。理解局部性，是写出高性能代码的基础。",
    "敦煌莫高窟的保护工作近日取得新进展。研究人员利用高精度三维扫描和多光谱成像，对部分壁画进行数字化采集，建立病害档案，并分析起甲、酥碱和褪色等问题的成因。洞窟环境监测显示，湿度与游客呼吸产生的水汽会加速壁画劣化，因此景区采取分时预约、限流参观和数字展示等方式平衡保护与利用。专家表示，数字化不是替代实体保护，而是为修复和传播提供依据。未来还需加强跨学科合作，让千年石窟在科学守护下延续。",
    "清晨六点半，菜市场比城市更早醒来。卖青菜的摊主把带着露水的菠菜码成小堆，鱼摊前的水声哗哗作响，豆腐铺飘出温热豆香。一位老人挑西红柿时反复捏了捏，又放下，转身去买更便宜的土豆。摊主们一边称重一边闲聊，谁家孩子开学了，哪条路又在修。这里没有宏大的叙事，只有斤两、零钱和熟客的招呼。太阳升高后，人流渐密，吆喝声此起彼伏。菜市场像一座城市的胃，朴素、嘈杂，却最知道日子真实的滋味。",
    "国产大模型推理成本持续下降引发关注。多家云厂商近期下调API价格，部分模型每百万token调用费用较年初明显降低。业内分析认为，成本下降来自模型结构优化、推理引擎改进和国产算力适配等多重因素。价格降低有助于推动智能客服、代码辅助和文档处理等应用落地，但也带来新问题：低价竞争可能压缩研发投入，企业需在性能、稳定性和成本之间找到平衡。专家建议，用户选型时不应只看单价，还要评估延迟、准确率和数据安全，避免被短期促销误导。",
    "老书店藏在巷子深处，木门推开时会发出一声低哑的吱呀。阳光从高窗斜落，照在积了薄尘的书脊上，空气里有纸张和旧木头混合的气味。店主坐在柜台后修一本脱线的诗集，针线穿过书脊，动作缓慢而稳。角落里有位年轻人翻着旧地图册，偶尔用铅笔在笔记本上描几笔。书店不卖咖啡，也不做直播，只在周末办小规模读书会。有人劝店主改造成网红空间，他笑笑说，书需要安静。傍晚关门时，他把门口小黑板擦净，重新写上：今日已开，明日亦然。",
            )

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
    llm.generate([chat_wrap(llm, "hi")], SamplingParams(temperature=TAU, max_tokens=4), use_tqdm=False)
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
    llm = build_llm(util=0.32)
    print(f"[diag] kv 池 {llm.model_runner.config.num_kvcache_blocks} 块 = "
          f"{llm.model_runner.config.num_kvcache_blocks * llm.model_runner.block_size} token 容量")
    sp = SamplingParams(temperature=TAU, max_tokens=MAX_TOKENS,ignore_eos=True)
    prompts = [chat_wrap(llm, f"把下面这句话原样重复 5 遍： {text}") for text in R1_TEXTS] * 2
    preempt_log = []
    orig_preempt = llm.scheduler.preempt
    def wrapped_preempt(seq):
        preempt_log.append(seq.num_spec_tokens)
        return orig_preempt(seq)
    llm.scheduler.preempt = wrapped_preempt
    bm = llm.scheduler.block_manager
    min_free = [len(bm.free_block_ids)]          # free 池最低水位：生成过程中块最紧张的时刻
    snap_done = [False]
    orig_step = llm.step
    def watching_step():
        result = orig_step()
        min_free[0] = min(min_free[0], len(bm.free_block_ids))
        seqs = list(llm.scheduler.running)
        if not snap_done[0] and seqs and min(len(s.block_table) for s in seqs) >= 8:
            snap_done[0] = True                  # 各条都长到 8 块时抓一次块表快照
            from collections import Counter
            tabs = [list(s.block_table) for s in seqs]
            total = sum(len(t) for t in tabs)
            union = set().union(*tabs)
            cnt = Counter(b for t in tabs for b in t)
            shared = sum(1 for v in cnt.values() if v > 1)
            print(f"[diag] 快照：{len(seqs)} 条 | 块表总项 {total} | 物理并集 {len(union)} | 被多条共享的块 {shared}")
            print(f"[diag] 各条块表长 {[len(t) for t in tabs]}")
        return result
    llm.step = watching_step
    out_small, cnt, _ = run_on(llm, prompts, sp)
    llm.step = orig_step
    llm.scheduler.preempt = orig_preempt
    print(f"preempt 次数 {len(preempt_log)}，各次 num_spec_tokens: {preempt_log}")
    print(f"  其中带草稿被抢（>0）的次数：{sum(1 for x in preempt_log if x > 0)}")
    print([len(o["token_ids"]) for o in out_small])
    print(f"[diag] free 块最低水位 {min_free[0]} / 池 {len(bm.blocks)}")
    atexit.unregister(llm.exit)
    llm.exit()
    torch.cuda.empty_cache()
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
    llm = build_llm(util=0.32)
    sp = SamplingParams(temperature=TAU, max_tokens=2048, ignore_eos=True)
    prompts = [chat_wrap(llm, f"把下面这段话原样重复 5 遍：{t}") for t in R1_TEXTS] * 2
    stats = {"calls": 0, "cached": 0, "demand": 0, "stale_probe": 0}
    bm = llm.scheduler.block_manager
    orig_probe = bm.can_allocate_prefill
    def wrapped_probe(seq):
        r = orig_probe(seq)
        stats["calls"] += 1
        stats["cached"] += max(0, r)
        stats["demand"] += seq.num_blocks
        return r
    orig_alloc = bm.allocate_block
    def wrapped_alloc(token_ids, h):
        # 量 B 档收益的真口径：满段才有本块键 = compute_hash(段, prefix)；
        # 键在表中且指向块内容 不符 = stale 白查。（入参 h 是前块 prefix，不是本块键——
        # 旧版 wrap 拿 prefix 比对当前段，误报 127 次，已修正 2026-10-21）
        if len(token_ids) == bm.block_size:
            key = BlockManager.compute_hash(token_ids, h)
            if key in bm.hash_to_block_id \
                    and bm.blocks[bm.hash_to_block_id[key]].token_ids != token_ids:
                stats["stale_probe"] += 1
                if stats["stale_probe"] <= 3:
                    bid = bm.hash_to_block_id[key]
                    blk = bm.blocks[bid]
                    print(f"[stale 现场] 键指向块 {bid} (ref={blk.ref_count}, "
                          f"hash字段指向自己={blk.hash == key})")
        return orig_alloc(token_ids, h)

    bm.can_allocate_prefill = wrapped_probe
    bm.allocate_block = wrapped_alloc
    run_on(llm, prompts, sp)
    bm.can_allocate_prefill = orig_probe
    bm.allocate_block = orig_alloc
    print(f"prefill 探测 {stats['calls']} 次，需求 {stats['demand']} 块，命中 {stats['cached']} 块")
    print(f"stale 白查 {stats['stale_probe']} 次 | 结束时 hash 表残留 {len(bm.hash_to_block_id)} 条")

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