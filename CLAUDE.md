# nano-vllm-specdec

## 1. 项目定位

在 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) v0.2.0（上游 commit `bb823b3`）基础上**增量实现** Speculative Decoding 的极简 LLM 推理引擎。

- **不是从零实现**。基线 commit `78d0719` 为未改动的上游代码，此后所有 commit 均为增量改动——任何对外表述（README / 简历）必须如实标注 based on nano-vllm
- 双路线：① n-gram proposer（greedy，第 1 步）→ ② draft-model（Qwen3-0.6B draft + Qwen3-1.7B target，temperature，第 2 步）
- 上游 #147 PR（未合并的 SpecDec 实现）只当**改动面地图**，不当模板

## 2. 架构指针

行数为 `wc -l` 实测（含空行/注释）。按"一个请求的生命周期"顺序阅读：

| 文件 | 行数 | 职责 |
|------|:---:|------|
| `nanovllm/llm.py` | 5 | `LLM.generate()` 入口，转发 engine |
| `nanovllm/engine/llm_engine.py` | 90 | 主循环：prefill/decode 交替，每步 add_req → schedule → run → postprocess |
| `nanovllm/engine/sequence.py` | 83 | Sequence 状态：token_ids / block_table / num_blocks / is_finished |
| `nanovllm/engine/scheduler.py` | 92 | 双队列（waiting/running）+ preempt 回滚 |
| `nanovllm/engine/block_manager.py` | 120 | KV cache 分块分配 + prefix caching（哈希 / 引用计数） |
| `nanovllm/engine/model_runner.py` | 257 | 调度结果变 tensor 喂模型：forward / sample / KV 写入 |
| `nanovllm/utils/context.py` | 27 | slot_mapping / block_tables 在层间传递的上下文 |
| `nanovllm/layers/attention.py` | 75 | attention 读 KV cache（计算外包 flash-attn） |
| `nanovllm/models/qwen3.py` | 216 | 模型结构（decoder-only，当前唯一支持） |
| `nanovllm/layers/` 其余 | — | linear / rotary / layernorm / sampler 等基础层 |
| `nanovllm/utils/loader.py`, `config.py`, `sampling_params.py` | — | 权重加载、引擎配置、采样参数 |

上游对照仓库：clone `GeeeekExplorer/nano-vllm` 到**本仓库外的独立目录**（各环境自行管理位置，勿写死路径）。diff 对照一律以它为准。

## 3. 红线（协作规则，最高优先级）

1. **核心代码全部由用户手写，每行能讲 3 分钟。** Claude 不代写引擎核心代码——只提供：小任务拆解、卡点解答、代码审查
2. **性能数字一律云卡产出**：开发调试 3080 Ti 12GB，简历数据 3090 24GB；本地 GTX 1650 4GB 只算编辑器，其数字不入任何记录
3. **数字纪律**：对照组同卡/同负载/同 seed、各 3 轮取均值；接受率 α（算法侧）与加速比（系统侧）分开报；加速比量级对照理论值 S=(α+1)/(1+kc) 自检，对不上即测错；每个数字必须能还原测量过程
4. **commit 有节奏**：一个小改动一个 commit，不批量 dump

## 4. 当前任务：第 1 步 n-gram 版 SpecDec（greedy）

改动面 8 文件，按依赖序实现（②③独立可单测，故前置）：

| 序 | 文件 | 改动 |
|:--:|------|------|
| ① | `engine/sequence.py` | spec_tokens / num_spec_tokens 字段 + merge_spec_tokens / extend_tokens |
| ② | `spec_decode/ngram_proposer.py`（新） | n-gram proposer：min/max_ngram 匹配 + k 三约束 |
| ③ | `layers/rejection_sampler.py`（新） | greedy 拒绝采样（argmax 比较版） |
| ④ | `engine/block_manager.py` | allocate_decode（预分配）/ deallocate_decode（回滚）+ **stale hash 防护**（#147 只重置字段没删字典映射——超越点） |
| ⑤ | `engine/scheduler.py` | schedule 返回 is_spec_decode；decode 分支 merge + allocate_decode；postprocess 多 token 确认 |
| ⑥ | `engine/model_runner.py` | prepare_spec_decode（varlen 打包）+ sample_drafts |
| ⑦ | `layers/attention.py` | is_spec_decode 分支走 flash_attn_varlen |
| ⑧ | `engine/llm_engine.py` | step() 串联四阶段（merge → verify → extend → sample_drafts） |

**验收**：`example.py` 跑通（enforce_eager）；greedy 下输出与不开 spec decode **逐 token 一致**；重复前缀负载下吞吐可见提升。

第 2 步（draft-model + temperature 拒绝采样 min(1, p/q) + 残差重采样）与第 3 步（benchmark + 简历数据）见外部主规划，不在本文件展开。

## 5. 协作模式

陪练四原则：**小任务驱动 / 先猜后跑 / 答案后置 / 只答卡点**。代码审查：贴 diff 或指文件路径。方向性建议必须带可复查引用（issue / PR / 实测）。
