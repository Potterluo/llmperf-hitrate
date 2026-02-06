# LLM 性能测试套件

针对 LLM 推理服务的性能基准工具，测量延迟（TTFT/TBT）、吞吐、并发稳定性及 Prefix Caching 效果。

## 功能特性

| 维度 | 说明 |
|------|------|
| **延迟分析** | TTFT、TBT、端到端延迟（P50/P90/P99） |
| **吞吐测试** | Total/Incremental Throughput |
| **并发压测** | 多并发度下的性能衰减曲线 |
| **缓存验证** | Prefix Cache Hit Rate 效果评估 |

## 配置说明

### 1. 服务端配置 (`config.yaml`)

```yaml
llm_connection:
  model: "Qwen3-32B"
  server_url: "http://141.111.32.70:8244"
  tokenizer_path: "/mnt/model/Qwen3-32B"
  llm_type: "vllm"  # 支持 vllm / sglang / mindie
  enable_clear_hbm: true
  timeout: 180
```

**Hit Rate 处理差异**：

| 引擎 | 策略 | 说明 |
|------|------|------|
| vLLM | 1次 prefill | 利用 Automatic Prefix Caching |
| SGLang | 2次 prefill | 双重预热确保 KV Cache 稳定 |
| MindIE | 缩短输入 | `input_tokens × hit_rate` 模拟命中 |

### 2. 负载参数 (`suites/E2E/test_uc_performance.py`)

```python
# 修改perf_scenarios字段，或配置环境变量 export PERF_TEST_CASE=[[4096, 1024, 1, 1, 0, 50]]
# (mean_in, mean_out, max_req, concurrent, random_seed, hit_rate)
perf_scenarios = [(100, 10, 1, 1, 0, 50)]
# 输入长度
# 输出长度
# 单轮请求数  
# 并发数，本case的所有并发，一般只跑一轮，等于单轮请求数 
# 随机种子0=随机, 其他=固定
# 缓存命中率(%)
```

## 快速开始

```bash
pip install -r requirements.txt

# 运行测试
pytest suites/E2E/test_uc_performance.py -v

# 或全量执行
pytest
```

## 结果指标

### 延迟 (s)
- `ttft_s`: 首 Token 延迟
- `inter_token_latency_s`: Token 间隔延迟  
- `end_to_end_latency_s`: 总延迟

### 吞吐 (tokens/s)
- `total_throughput`: 总吞吐
- `incremental_throughput`: 解码阶段吞吐

### 输出示例
```python
{
    "model": "Qwen3-32B",
    "timestamp": 1738321800,
    "results_ttft_s_mean": 0.125,
    "results_ttft_s_p99": 0.245,
    "results_total_throughput": 245.8,
    "results_num_completed_requests": 100
}
```

支持输出到 JSONL 和 CSV（通过 `config.yaml` 的 `database` 字段配置）。
```yaml
database:
  backup: "results/"
```
