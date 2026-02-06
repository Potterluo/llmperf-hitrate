import json
import logging
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Global state
_backup_path: Optional[Path] = None
_test_build_id: Optional[str] = None


def _get_backup_config() -> Dict[str, Any]:
    """获取备份配置（不再依赖数据库配置）"""
    try:
        from common.config_utils import config_utils as config_instance
        config = config_instance.get_config("database", {})
        return {
            "backup": config.get("backup", "results/"),
            "enabled": True  # 始终启用文件写入
        }
    except Exception as e:
        logger.warning(f"Failed to load config, using defaults: {e}")
        return {"backup": "results/", "enabled": True}


def _initialize_backup_path() -> Path:
    """初始化并返回备份目录路径"""
    global _backup_path
    if _backup_path is not None:
        return _backup_path

    config = _get_backup_config()
    backup_str = config.get("backup", "results/")
    _backup_path = Path(backup_str).resolve()
    _backup_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Backup directory set to: {_backup_path}")
    return _backup_path


def _set_test_build_id(build_id: Optional[str] = None) -> None:
    global _test_build_id
    _test_build_id = build_id or f"build_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    logger.debug(f"Test build ID set to: {_test_build_id}")


def _get_test_build_id() -> str:
    global _test_build_id
    if _test_build_id is None:
        _set_test_build_id()
    return _test_build_id


def _write_to_jsonl(table_name: str, data: Dict[str, Any]) -> bool:
    """
    将数据追加写入 JSONL 文件
    
    Args:
        table_name: 用作文件名前缀（如 "test_results" → test_results.jsonl）
        data: 要写入的字典数据
    
    Returns:
        bool: 写入是否成功
    """
    backup_path = _initialize_backup_path()
    file_path = backup_path / f"{table_name}.jsonl"
    
    try:
        # 添加通用字段
        record = data.copy()
        record.setdefault("id", datetime.utcnow().strftime("%Y%m%d%H%M%S%f"))
        record.setdefault("created_at", datetime.utcnow().isoformat())
        record.setdefault("test_build_id", _get_test_build_id())
        
        # 写入 JSONL（每行一个 JSON 对象）
        with file_path.open("a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, default=str)
            f.write("\n")
        
        logger.info(f"Data written to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write to {file_path}: {e}", exc_info=True)
        return False


def write_to_file(table_name: str, data: Dict[str, Any]) -> bool:
    """
    主写入接口：将数据写入 JSONL 文件
    
    Args:
        table_name: 表名（用作文件名）
        data: 要写入的数据字典
    
    Returns:
        bool: 写入是否成功
    """
    return _write_to_jsonl(table_name, data)


def jsonl_to_csv(
    jsonl_path: Union[str, Path],
    csv_path: Optional[Union[str, Path]] = None,
    flatten: bool = False
) -> Path:
    """
    将 JSONL 文件转换为 CSV 文件
    
    Args:
        jsonl_path: JSONL 文件路径
        csv_path: 可选，输出 CSV 路径（默认与 JSONL 同目录同名 .csv）
        flatten: 是否展平嵌套字典（使用点号表示法，如 "user.name"）
    
    Returns:
        Path: 生成的 CSV 文件路径
    
    Example:
        jsonl_to_csv("results/test.jsonl")
        # 生成 results/test.csv
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    if csv_path is None:
        csv_path = jsonl_path.with_suffix(".csv")
    else:
        csv_path = Path(csv_path)
    
    # 读取所有 JSONL 记录
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if flatten:
                    record = _flatten_dict(record)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    
    if not records:
        raise ValueError(f"No valid records found in {jsonl_path}")
    
    # 收集所有字段名（保持插入顺序）
    fieldnames = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    
    # 写入 CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    
    logger.info(f"Converted {len(records)} records to {csv_path}")
    return csv_path


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    递归展平嵌套字典
    
    Example:
        {"a": {"b": 1}} → {"a.b": 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def set_build_id(build_id: str) -> None:
    """设置构建 ID（供外部调用）"""
    _set_test_build_id(build_id)
    logger.info(f"Build ID set to: {build_id}")


# 兼容旧接口（可选）
write_to_db = write_to_file
def database_connection(build_id):
    """模拟数据库连接函数，实际不连接数据库"""
    logger.info("Database connection simulated (no actual DB connection).")
    set_build_id(build_id)
    return None