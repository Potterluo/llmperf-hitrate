import hashlib
import json
import logging
import math
import os
import pathlib
import random
import subprocess
import time
from typing import Any, Dict, Tuple

from common.config_utils import config_utils
from transformers import LlamaTokenizerFast

# 配置模块级日志记录器
logger = logging.getLogger(__name__)

RESULTS_VERSION = "2025-10-30"


def get_clear_hbm_config() -> bool:
    """延迟获取配置，避免模块导入时配置未就绪"""
    return config_utils.get_nested_config("llm_connection.enable_clear_hbm", True)


class LLMPerfResults:
    """LLM性能测试结果封装类"""
    
    def __init__(
        self,
        name: str,
        metadata: Dict[str, Any] | None = None,
    ):
        self.name = name
        self.metadata = metadata if metadata is not None else {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """转换为扁平化字典"""
        data = {
            "version": self.version,
            "name": self.name,
        }
        data.update(self.metadata)
        return flatten_dict(data)

    def json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


def randomly_sample_sonnet_lines_prompt(
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    tokenizer: LlamaTokenizerFast | None = None,
) -> Tuple[str, int]:
    """随机采样十四行诗文本，生成指定token长度的prompt
    
    Args:
        prompt_tokens_mean: prompt token数量的均值
        prompt_tokens_stddev: prompt token数量的标准差
        tokenizer: 分词器实例
        
    Returns:
        Tuple[str, int]: 生成的prompt文本和实际token数量
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    
    def get_token_length(text: str) -> int:
        return len(tokenizer.encode(text))

    base_prompt = (
        "Randomly stream lines from the following text "
        "Don't generate eos tokens:\n\n"
    )
    
    # 确保prompt长度至少为基础prompt的长度
    base_length = get_token_length(base_prompt)
    num_prompt_tokens = sample_random_positive_int(prompt_tokens_mean, prompt_tokens_stddev)
    while num_prompt_tokens < base_length:
        num_prompt_tokens = sample_random_positive_int(prompt_tokens_mean, prompt_tokens_stddev)
    
    remaining_tokens = num_prompt_tokens - base_length
    
    # 读取十四行诗文件
    sonnet_path = pathlib.Path(__file__).parent.resolve() / "sonnet.txt"
    try:
        with open(sonnet_path, "r", encoding="utf-8") as f:
            sonnet_lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"Sonnet file not found: {sonnet_path}")
        raise
    except IOError as e:
        logger.error(f"Failed to read sonnet file: {e}")
        raise
    
    random.shuffle(sonnet_lines)
    
    # 构建prompt
    prompt = base_prompt
    for line in sonnet_lines:
        line_tokens = get_token_length(line)
        if remaining_tokens - line_tokens < 0:
            # 截断最后一行以适应精确的token数量
            # 注意：这会截断单词，但LLM应该能处理
            chars_to_take = int(math.ceil(remaining_tokens))
            prompt += line[:chars_to_take]
            break
        prompt += line
        remaining_tokens -= line_tokens
    
    # 记录prompt哈希用于调试/验证
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    logger.debug(f"Generated prompt hash: {prompt_hash[:16]}... (length: {num_prompt_tokens} tokens)")
    
    return prompt, num_prompt_tokens


def sample_random_positive_int(mean: int, stddev: int) -> int:
    """从高斯分布中采样正整数
    
    持续采样直到获得正数结果
    
    Args:
        mean: 高斯分布均值
        stddev: 高斯分布标准差
        
    Returns:
        随机正整数
    """
    if stddev < 0:
        raise ValueError("stddev must be non-negative")
    
    # 避免无限循环：如果mean为负数且绝对值很大，可能很难采样到正数
    # 设置最大尝试次数
    max_attempts = 10000
    attempts = 0
    
    while attempts < max_attempts:
        value = int(random.gauss(mean, stddev))
        if value > 0:
            return value
        attempts += 1
    
    # 如果多次尝试失败，返回1作为安全默认值
    logger.warning(f"Failed to sample positive int after {max_attempts} attempts, returning 1")
    return 1


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """扁平化嵌套字典
    
    Args:
        d: 待扁平化的字典
        parent_key: 父键前缀
        sep: 键分隔符
        
    Returns:
        扁平化后的字典
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reset_prefill_cache(env: Dict[str, str], server_url: str, llm_type: str) -> None:
    """重置模型前缀缓存/HBM
    
    通过HTTP端点清除vLLM或SGLang的KV缓存
    
    Args:
        env: 环境变量字典
        server_url: 服务URL
        llm_type: 模型类型 (vllm/sglang)
    """
    if not get_clear_hbm_config():
        logger.debug("Clear HBM is disabled, skipping cache reset")
        return

    # 根据模型类型确定端点
    endpoint_map = {
        "vllm": "/reset_prefix_cache",
        "sglang": "/flush_cache",
    }
    
    if llm_type not in endpoint_map:
        raise ValueError(f"Unsupported llm_type: {llm_type}. Supported: {list(endpoint_map.keys())}")
    
    reset_url = f"{server_url}{endpoint_map[llm_type]}"
    logger.info(f"Resetting prefix cache: {reset_url}")

    try:
        result = subprocess.run(
            ["curl", "-X", "POST", reset_url, "-s", "-f"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        
        if result.returncode == 0:
            logger.info("Prefix cache successfully reset")
        else:
            stderr = result.stderr.strip() if result.stderr else "No error message"
            logger.warning(
                f"Failed to reset prefix cache. "
                f"Exit code: {result.returncode}, "
                f"URL: {reset_url}, "
                f"Error: {stderr}"
            )
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout while resetting prefix cache (10s): {reset_url}")
    except FileNotFoundError:
        logger.warning("curl command not found in PATH")
    except subprocess.SubprocessError as e:
        logger.warning(f"Subprocess error during cache reset: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error during cache reset: {e}", exc_info=True)