# nano-vllm-specdec

**在 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（约 1200 行的极简 LLM 推理引擎）上实现 n-gram 投机解码（Speculative Decoding）**。

草稿 token 由对已生成文本的模式匹配免费提出，一次前向批量验证，KV cache 被抢占时安全回滚；并带上游所禁止的 greedy（`temperature=0`）安全入口。

## 原理：为什么能加速

LLM 解码阶段是 **memory-bound** 的：每生成一个 token 都要读一遍全部模型权重，算力大量闲置。投机解码正是利用这一点——用极低代价提出 `k` 个草稿 token（本仓：对上下文做最长后缀 n-gram 匹配，无需额外模型），**一次**前向同时验证，代替 `k` 次串行解码；验证通过的 token 整批提交。文本可预测时，一轮最多产出 `k+1` 个 token；不可预测时，该轮优雅退化为 1 个 token，不产生错误输出。

## 改动面（相对上游 nano-vllm）

| 组件 | 文件 | 职责 |
|---|---|---|
| n-gram 起草器 | `nanovllm/spec_decode/ngram_proposer.py` | 最长后缀匹配提出草稿（KMP/LPS），`k=4` |
| 拒绝采样器 | `nanovllm/layers/rejection_sampler.py` | 逐位草稿验证；`temperature=0` 时逐位分流 greedy argmax |
| 投机感知的 KV 管理 | `nanovllm/engine/block_manager.py` | 每轮 `k` 个草稿 token 的预分配与回滚 |
| 投机轮调度 | `nanovllm/engine/scheduler.py` | verify 轮调度；被抢占时的草稿回滚（`unmerge_spec_tokens`） |
| Model runner | `nanovllm/engine/model_runner.py` | varlen 投机前向（每序列 `k+1` token）、逐位温度展开 |
| 引擎接线 | `llm_engine.py` / `sequence.py` / `utils/context.py` / `layers/attention.py` | 投机状态（`decode_*` 属性族）、context 传递、varlen attention 分支 |
| 采样参数 | `nanovllm/sampling_params.py` | 放行 `temperature=0`（上游 `assert > 1e-10` 禁止） |

## 性能对比

**配置**：RTX 3080 Ti（12GB, Ampere）· Qwen3-0.6B · `k=4` · `enforce_eager=True` · 每配置单轮测量（已做 warmup 预热排除首次编译开销；与独立的 monkey-patch 实测交叉验证，详见 `docs/benchmark.md`）。

| 负载 | 关投机 | 开投机 | 加速比 | spec 轮占比 | 接受 token 数 → 轮数 |
|---|---|---|---|---|---|
| 逐字复述 | 25.2 tok/s | 51.4 tok/s | **x2.04** | 14/18 | {5:12, 4:1, 1:5} |
| 数数（伪模式） | 27.5 tok/s | 26.7 tok/s | x0.97 | 104/185 | {1:182, 2:2, 5:1} |
| 开放生成 | 29.4 tok/s | 29.1 tok/s | x0.99 | 1/59 | {1:59} |

真重复负载约翻倍；**数数与开放生成持平到微亏**——n-gram 草稿只有当文本真的重复时才会被接受（"1, 2, 3…"看着像模式，但自身从不重复）。负例结果如实报告是有意的：知道投机在哪儿**不**划算，是结论的一部分。

`k=4` 为默认配置；完整 k 扫描列为后续工作（见 `docs/benchmark.md`）。

## 正确性验证

- **greedy 逐 token 等价**：开 / 关投机两次生成，输出逐 token 完全一致——两类 prompt 共 70 / 192 token（`compare_greedy.py`）
- **`temperature=0` 路径**：与近零温度（1e-6）在双 case 上输出全等——两条 greedy 路径互证
- **显存压力下**（16 条并发、KV free 块压到 0）：抢占 11 次、回滚后零草稿残留、全部输出完好（`spec_bench.py r1`）

## 快速开始

```bash
pip install git+https://github.com/xiaopei-1/nano-vllm-specdec.git
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B", enforce_eager=True)
sampling_params = SamplingParams(temperature=0, max_tokens=256)   # greedy 现在可用了
outputs = llm.generate(["你好，投机解码。"], sampling_params)
print(outputs[0]["text"])
```

## 复现

```bash
python compare_greedy.py          # greedy 等价性：开/关投机逐 token 对照
python spec_bench.py bench        # 三负载吞吐表（上方数据）
python spec_bench.py r1           # 抢占 × 草稿回滚压力测试（16 条并发 + 紧 KV 池）
python spec_bench.py hash         # stale hash 量化（前缀缓存卫生）

pytest tests/                     # 24 个单元测试
```

## 与上游的关系

- Fork 自 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（v0.2.0 基线）。上游明确不规划投机解码（[#14](https://github.com/GeeeekExplorer/nano-vllm/issues/14)），本仓即为此方向的实验场。
- 该生态唯一的先行尝试（[#147](https://github.com/GeeeekExplorer/nano-vllm/pull/147)，未合并）为本仓参照。相对它修复了 5 处缺陷：缺 `max_model_len` 回退、多序列比较错位、恰满末块 hash 误重置、抢占 × 草稿合并残留、stale hash 删除守卫。明细见 `docs/benchmark.md`。
