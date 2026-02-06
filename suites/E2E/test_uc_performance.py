import json
import os

import pytest
from common.capture_utils import export_vars
from common.config_utils import config_utils as config_instance
from common.llmperf.run_inference import inference_results

perf_scenarios = [
    # (mean_in, mean_out, max_req, concurrent, random_seed, hit_rate)
    (100, 10, 1, 1, 0, 50)
]
perf_test_case_str = os.getenv("PERF_TEST_CASE")
if perf_test_case_str:
    try:
        parsed = json.loads(perf_test_case_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            valid = True
            result = []
            for item in parsed:
                if not isinstance(item, (list, tuple)) or len(item) != 6:
                    valid = False
                    break
                try:
                    result.append(tuple(int(x) for x in item))
                except (ValueError, TypeError):
                    valid = False
                    break       
            if valid:
                perf_scenarios = result
                print(f"成功从环境变量加载配置: {perf_scenarios}")
            else:
                print("环境变量格式无效，使用默认配置")
        else:
            print("环境变量解析结果为空或非列表，使用默认配置")
            
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}，使用默认配置")
    except Exception as e:
        print(f"解析异常: {e}，使用默认配置")
else:
    print("未设置 PERF_TEST_CASE 环境变量，使用默认配置")
print(f"最终 perf_scenarios: {perf_scenarios}")

scenario_ids = [f"in_{s[0]}-out_{s[1]}-con_{s[3]}" for s in perf_scenarios]
TOTAL_COUNTER = len(perf_scenarios)
ROUND_COUNTER = 1


@pytest.mark.stage(2)
@pytest.mark.feature("uc_performance_test")
@pytest.mark.parametrize(
    "in_tokens, out_tokens, max_req, concurrent, random_seed, hit_rate",
    perf_scenarios,
    ids=scenario_ids,
)
@export_vars
def test_performance(in_tokens, out_tokens, max_req, concurrent, random_seed, hit_rate):
    global TOTAL_COUNTER
    global ROUND_COUNTER
    summary = inference_results(
        [in_tokens],
        [out_tokens],
        [max_req],
        [concurrent],
        [random_seed],
        [hit_rate],
        TOTAL_COUNTER,
        ROUND_COUNTER,
    )
    ROUND_COUNTER += 1
    results = summary.get("results", {})

    import math
    import os

    # 如果 summary 为空（None 或 {}），直接返回全 nan 的字典
    if not summary:
        metrics = {
            "input_tokens": float('nan'),
            "output_tokens": float('nan'),
            "concurrent": float('nan'),
            "sum_requests": float('nan'),
            "hit_rate": float('nan'),
            "ttft_mean": float('nan'),
            "tpot_mean": float('nan'),
            "total_throughput": float('nan'),
            "e2e_mean": float('nan'),
            "extra_info": os.getenv("TEST_EXTRA_INFO")
            or config_instance.get_nested_config("llm_connection.extra_info"),
            "model": config_instance.get_nested_config("llm_connection.model"),
            "server_url": config_instance.get_nested_config("llm_connection.server_url"),
            "tbt_p50": float('nan'),
            "tbt_p90": float('nan'),
            "tbt_p99": float('nan'),
            "ttft_p50": float('nan'),
            "ttft_p90": float('nan'),
            "ttft_p99": float('nan'),
            "e2e_p50": float('nan'),
            "e2e_p90": float('nan'),
            "e2e_p99": float('nan'),
            "num_completed_requests": float('nan'),
            "elapsed_time": float('nan'),
            "incremental_throughput": float('nan'),
        }
    else:
        results = summary.get("results", {}) or {}

        metrics = {
            # 输入指标
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "concurrent": concurrent,
            "sum_requests": max_req,
            "hit_rate": hit_rate,
            
            # 延迟指标（带空值保护）
            "ttft_mean": results.get("ttft_s", {}).get("mean") if results.get("ttft_s") else float('nan'),
            "tpot_mean": results.get("inter_token_latency_s", {}).get("mean") if results.get("inter_token_latency_s") else float('nan'),
            "total_throughput": summary.get("total_throughput") if summary.get("total_throughput") is not None else float('nan'),
            "e2e_mean": results.get("end_to_end_latency_s", {}).get("mean") if results.get("end_to_end_latency_s") else float('nan'),
            
            # 模型及环境信息
            "extra_info": os.getenv("TEST_EXTRA_INFO")
            or config_instance.get_nested_config("llm_connection.extra_info"),
            "model": config_instance.get_nested_config("llm_connection.model"),
            "server_url": config_instance.get_nested_config("llm_connection.server_url"),
            
            # TPOT 分位数
            "tbt_p50": (results.get("inter_token_latency_s", {}).get("quantiles") or {}).get("p50") if results.get("inter_token_latency_s") else float('nan'),
            "tbt_p90": (results.get("inter_token_latency_s", {}).get("quantiles") or {}).get("p90") if results.get("inter_token_latency_s") else float('nan'),
            "tbt_p99": (results.get("inter_token_latency_s", {}).get("quantiles") or {}).get("p99") if results.get("inter_token_latency_s") else float('nan'),
            
            # TTFT 分位数
            "ttft_p50": (results.get("ttft_s", {}).get("quantiles") or {}).get("p50") if results.get("ttft_s") else float('nan'),
            "ttft_p90": (results.get("ttft_s", {}).get("quantiles") or {}).get("p90") if results.get("ttft_s") else float('nan'),
            "ttft_p99": (results.get("ttft_s", {}).get("quantiles") or {}).get("p99") if results.get("ttft_s") else float('nan'),
            
            # E2E 分位数
            "e2e_p50": (results.get("end_to_end_latency_s", {}).get("quantiles") or {}).get("p50") if results.get("end_to_end_latency_s") else float('nan'),
            "e2e_p90": (results.get("end_to_end_latency_s", {}).get("quantiles") or {}).get("p90") if results.get("end_to_end_latency_s") else float('nan'),
            "e2e_p99": (results.get("end_to_end_latency_s", {}).get("quantiles") or {}).get("p99") if results.get("end_to_end_latency_s") else float('nan'),
            
            # 吞吐统计
            "num_completed_requests": summary.get("num_completed_requests") if summary.get("num_completed_requests") is not None else float('nan'),
            "elapsed_time": summary.get("elapsed_time") if summary.get("elapsed_time") is not None else float('nan'),
            "incremental_throughput": summary.get("incremental_throughput") if summary.get("incremental_throughput") is not None else float('nan'),
        }

    for key, val in metrics.items():
        assert val is not None, f"Metric '{key}' is missing"

    return {"_name": "llmperf", "_proj": metrics}
