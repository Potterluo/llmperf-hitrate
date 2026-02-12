# LLM-Perf 性能测试操作手册

本工具用于测试 LLM 推理服务的性能，包括：

- 延迟
- 吞吐
- 并发稳定性
- Prefix Cache 命中效果
- UCM 纯盘命中场景验证（MindIE）

---

本工具支持两种运行方式：

| 方式            | 适合人群                   |
| --------------- | -------------------------- |
| Python 本地运行 | 有 Python 3.11+ 且能联网   |
| Docker 镜像运行 | 无法联网 / Python 版本不够 |

------

# 方式一：Python 本地运行

## 1️⃣ 下载代码

进入 GitHub 页面：

```
https://github.com/Potterluo/llmperf-hitrate
```

点击右上角：

```
Code → Download ZIP
```

下载后解压。

或者命令行下载：

```bash
git clone https://github.com/Potterluo/llmperf-hitrate.git
cd llmperf-hitrate
```

------

## 2️⃣ 安装依赖

确保系统 Python ≥ 3.11

```bash
python3 --version
```

安装依赖：

```bash
pip install -r requirements.txt
```

------

## 3️⃣ 修改服务配置

打开：

```
config.yaml
```

修改以下字段：

```yaml
llm_connection:
  model: "Qwen3-32B"
  server_url: "http://141.111.32.70:8244"
  tokenizer_path: "/mnt/model/Qwen3-32B"
  llm_type: "mindie"   # vllm / sglang / mindie
  enable_clear_hbm: true
  timeout: 180
```

### 参数说明（只需要填对）

| 参数           | 填什么              |
| -------------- | ------------------- |
| model          | 服务端模型名        |
| server_url     | 推理服务地址        |
| tokenizer_path | 模型 tokenizer 路径 |
| llm_type       | 必须和服务类型一致  |

------

## 4️⃣ 修改测试参数

打开：

```
suites/E2E/test_uc_performance.py
```

找到：

```python
perf_scenarios = [(100, 10, 1, 1, 0, 50)]
```

格式：

```
(mean_in, mean_out, max_req, concurrent, random_seed, hit_rate)
```

示例：

```python
perf_scenarios = [
    (4096, 1024, 1, 1, 0, 80)
]
```

含义：

| 位置 | 含义                |
| ---- | ------------------- |
| 4096 | 输入长度            |
| 1024 | 输出长度            |
| 1    | 请求数 (理解为并发) |
| 1    | 并发                |
| 0    | 随机种子            |
| 80   | 命中率              |

------

## 5️⃣ 运行测试

```bash
pytest suites/E2E/test_uc_performance.py -v
```

或者全部执行：

```bash
pytest
```

------

# 方式二：Docker 镜像运行（推荐）

适用于：

- 无法联网
- Python 版本低
- 需要统一环境

------

## 1️⃣ 下载镜像

进入发布页面：

```
https://github.com/Potterluo/llmperf-hitrate/releases/tag/0.0.1-release
```

根据机器架构下载：

| 机器类型  | 下载文件           |
| --------- | ------------------ |
| x86服务器 | llmperf-x86_64.tar |
| ARM服务器 | llmperf-arm64.tar  |

------

## 2️⃣ 加载镜像

```bash
docker load -i llmperf-x86_64.tar
```

------

## 3️⃣ 启动测试

假设：

- 模型路径：`/mnt/model`
- 本地代码路径：`/mnt/d/Project/llmperf`

运行：

```bash
docker run --rm -it \
  -v /mnt/model:/mnt/model-v \
  -v /mnt/d/Project/llmperf:/workspace \
  -w /workspace \
  llmperf:x86_64 pytest
```

说明：

| 参数   | 含义         |
| ------ | ------------ |
| -v     | 映射本地目录 |
| -w     | 工作目录     |
| pytest | 启动测试     |

------

## 后台运行

```bash
docker run -d --name=perf-test \
  -v /mnt/model:/mnt/model-v \
  -v /mnt/d/Project/llmperf:/workspace \
  -w /workspace \
  llmperf:x86_64 pytest
```

查看日志：

```bash
docker logs -f perf-test
```

保存日志：

```bash
docker logs perf-test > output.log
```

删除容器：

```bash
docker rm -f perf-test
```

------

# 三、测试结果获取

默认结果会保存在：

```
results/
```

可以在 config.yaml 中设置：

```yaml
database:
  backup: "results/"
```

运行完成后会生成：

- `.jsonl`
- `.csv`

示例：

```
results/llmperf.csv
results/llmperf.jsonl
```

------

# 四、不同引擎命中策略说明

## vLLM

- 自动 prefix cache
- 只需设置 hit_rate
- 不需要重启服务

------

## SGLang

- 会做双重 prefill
- 不需要手动干预

------

## MindIE

MindIE 的纯盘命中需要特殊流程，否则测试结果不真实。

原因：

如果不重启服务，显存中的 KV Cache 会优先命中，而不是磁盘缓存。

- 步骤 1：预热注入

修改测试参数：

```python
perf_scenarios = [(4096, 1, 1, 1, 95273856, 80)]
```

运行测试，发送 80% 长度的请求，把 KV Cache 写入 UCM 磁盘。

- 步骤 2：重启 MindIE 服务

目的：清空显存 KV Cache，仅保留磁盘数据

- 步骤 3：正式压测

修改测试参数：

```python
perf_scenarios = [(4096, 1024, 1, 1, 95273856, 100)]
```

运行：测到的性能是真实磁盘命中性能；保证第5个参数前后一致才能获取与预埋时一致的输入数据。

