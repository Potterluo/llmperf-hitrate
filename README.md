# LLM-Perf 性能测试操作手册

本工具用于测试 **LLM 推理服务**的性能，兼容 OpenAI 接口（`/v1/chat/completions`）。

主要测量指标：

- **延迟**：TTFT（首 Token 延迟）、TPOT（每 Token 生成延迟）、TBT（Token 间延迟）、端到端延迟
- **吞吐**：单请求吞吐、整体吞吐
- **并发稳定性**：多并发下的延迟分位数（p50/p90/p99）
- **Prefix Cache 命中效果**：通过命中率参数构造可复现的命中场景
- **UCM 纯盘命中场景验证**（MindIE 专用）

## 路线图

- [x] 基础功能
- [x] PrefTest 集成功能
- [x] TPOT 指标统计
- [ ] EvalTest 测试
- [ ] 配置文件工具增强
- [ ] 多种数据保存方式
- [ ] 统一迁移 llm_connection

---

# 方式一：本地运行（有网环境）

适合：能联网、Python ≥ 3.11 的机器。直接 `pytest` 运行，最简单。

## 1. 下载代码

```bash
git clone https://github.com/Potterluo/llmperf-hitrate.git
cd llmperf-hitrate
```

> 也可以在 GitHub 页面点击 `Code → Download ZIP` 下载解压。

## 2. 安装依赖

确认 Python 版本 ≥ 3.11：

```bash
python --version
```

安装依赖：

```bash
pip install -r requirements.txt
```

验证安装成功：

```bash
python -c "import pytest, pandas, transformers, requests; print('OK')"
```

看到 `OK` 即可。

## 3. 修改服务配置

打开项目根目录下的 `config.yaml`，找到 `llm_connection` 段，修改以下 4 个**必改**项：

```yaml
llm_connection:
  model: "Qwen3-32B"                    # 【必改】模型名，需与服务端一致
  server_url: "http://192.168.1.10:8000" # 【必改】推理服务地址（无需带 /v1）
  tokenizer_path: "/mnt/model/Qwen3-32B" # 【必改】本地 tokenizer 路径
  llm_type: "vllm"    # 【必改】引擎类型: vllm / sglang / mindie
  enable_clear_hbm: true   # 每轮请求前重置 prefix cache（建议 true）
  stream: true    # 流式输出（建议 true，否则无法测 TTFT/TPOT）
  ignore_eos: true    # 忽略结束符（压测时建议 true，保证固定输出长度）
  timeout: 180    # 单请求超时（秒）
  verbose: false   # 详细日志：开启后保存每次请求/响应原文到 results/verbose/
  extra_info: ""  # 标记信息，用于区分不同服务拉起参数
```

| 参数             | 填什么                       | 示例                       |
| ---------------- | ---------------------------- | -------------------------- |
| `model`          | 服务端模型名                 | `Qwen3-32B`                |
| `server_url`     | 推理服务地址                 | `http://192.168.1.10:8000` |
| `tokenizer_path` | 本地 tokenizer 路径          | `/mnt/model/Qwen3-32B`     |
| `llm_type`       | 引擎类型（必须和服务一致）   | `vllm` / `sglang` / `mindie` |

> 💡 **tokenizer 怎么找？** 就是模型目录下包含 `tokenizer.json` / `tokenizer_config.json` 的那一层目录，通常就是模型权重所在目录。

## 4. 修改测试参数

打开 `suites/E2E/test_uc_performance.py`，找到 `perf_scenarios`：

```python
perf_scenarios = [
    # (mean_in, mean_out, max_req, concurrent, random_seed, hit_rate)
    (100, 10, 1, 1, 0, 50)
]
```

格式说明：

| 位置 | 字段          | 含义                                    | 示例       |
| ---- | ------------- | --------------------------------------- | ---------- |
| 1    | `mean_in`     | 输入 token 长度                         | `4096`     |
| 2    | `mean_out`    | 输出 token 长度                         | `1024`     |
| 3    | `max_req`     | 请求总数                                | `10`       |
| 4    | `concurrent`  | 并发数                                  | `1`        |
| 5    | `random_seed` | 随机种子（0 = 随机，固定值 = 可复现）   | `95273856` |
| 6    | `hit_rate`    | 命中率（0-100，0 = 不测命中，详见下文） | `80`       |

> ⚠️ **命中率说明**：`hit_rate` 控制 prefix cache 命中比例。`0` 表示不测命中（纯打延迟/吞吐）；`80` 表示先用 `mean_in × 80%` 长度的 prompt 预热注入，再用完整 `mean_in` 的 prompt 压测，使前 80% 命中缓存。

**示例：测一组 4096 输入 / 1024 输出 / 并发 1 / 命中率 80%**

```python
perf_scenarios = [
    (4096, 1024, 1, 1, 0, 80)
]
```

**示例：测多组场景**（每行一组，独立出报告）

```python
perf_scenarios = [
    (100, 10, 1, 1, 0, 0),      # 短输入不测命中
    (4096, 1024, 1, 1, 0, 80),   # 长输入 80% 命中
]
```

## 5. 运行测试

```bash
pytest suites/E2E/test_uc_performance.py -v
```

或运行全部用例：

```bash
pytest
```

## 6. 查看结果

结果默认保存在项目根目录的 `results/` 下：

| 文件                    | 内容                            |
| ----------------------- | ------------------------------- |
| `results/llmperf.jsonl` | 每条测试记录（JSON Lines 格式） |
| `results/llmperf.csv`   | 同内容 CSV，方便 Excel 打开     |

各字段含义见文末 [测试结果字段说明](#测试结果字段说明)。

> 💡 **开 verbose 看请求原文**：把 `config.yaml` 里 `verbose` 改为 `true`，每次请求/响应会额外保存到 `results/verbose/exchange_*.json`，方便排查"为什么这次延迟异常"。

---

# 方式二：Docker 镜像运行

适合：无法联网 / Python 版本不够 / 需要统一环境的机器。用发布好的镜像运行，开箱即用。

镜像里**不含**代码和模型，需要通过 Docker 挂载把它们引入容器。下面按顺序操作。

## 1. 下载镜像

进入发布页面：<https://github.com/Potterluo/llmperf-hitrate/releases>

根据**机器架构**和**模型类型**下载对应镜像文件：

| 机器类型   | 非 GLM-5 模型（transformers 4.57.6） | GLM-5 系列模型（transformers 5.2.0） |
| ---------- | ----------------------------------- | ----------------------------------- |
| x86 服务器 | `llmperf-x86_64-tf4.tar`            | `llmperf-x86_64-tf5.tar`            |
| ARM 服务器 | `llmperf-arm64-tf4.tar`             | `llmperf-arm64-tf5.tar`             |

> ⚠️ **怎么选？** 测 GLM-5 系列模型选 `tf5`；测其他模型（Qwen、Llama、DeepSeek 等）选 `tf4`。原因见文末 [FAQ](#faq)。

## 2. 加载镜像

以 ARM 服务器 + 非 GLM-5 模型为例（请把文件名换成你下载的那个）：

```bash
docker load -i llmperf-arm64-tf4.tar
```

加载后会得到镜像 `llmperf:arm64-tf4`。镜像名规则：`llmperf:{架构}-tf{版本}`。

## 3. 下载代码

镜像不含代码，需要先拿到源码：

```bash
git clone https://github.com/Potterluo/llmperf-hitrate.git
cd llmperf-hitrate
```

> 也可以在 GitHub 页面 `Code → Download ZIP` 下载解压。后面所有配置修改都在这份代码里改。

## 4. 理解挂载路径与配置的关系（重要）

Docker 运行时，容器看不到宿主机的文件，必须用 `-v 宿主机路径:容器路径` 把目录"映射"进去。`config.yaml` 里填的路径必须是**容器内能看到的路径**，不是宿主机路径。

下面用一组明确的例子说明。假设宿主机上：

- 代码在：`/mnt/d/Project/llmperf`（你刚 clone 的目录）
- 模型在：`/mnt/model/Qwen3-32B`（tokenizer 文件所在目录）

我们要做两个挂载：

| 挂载什么 | 命令片段                            | 容器内对应路径        | 作用                       |
| -------- | ----------------------------------- | --------------------- | -------------------------- |
| 代码目录 | `-v /mnt/d/Project/llmperf:/workspace` | `/workspace`          | 容器读代码、写结果都在这里 |
| 模型目录 | `-v /mnt/model:/mnt/model`          | `/mnt/model`          | 容器读 tokenizer           |

> 💡 **关键技巧**：模型目录挂载时，让容器内路径和宿主机路径保持一致（都写 `/mnt/model`），这样 `config.yaml` 里的 `tokenizer_path` 直接写 `/mnt/model/Qwen3-32B` 就行，不用换算路径。

## 5. 修改配置文件

在**宿主机的代码目录**里改 `config.yaml`（改完容器会直接读到，因为是挂载的）。找到 `llm_connection` 段：

```yaml
llm_connection:
  model: "Qwen3-32B"                       # 【必改】模型名，需与服务端一致
  server_url: "http://127.0.0.1:8000"      # 【必改】推理服务地址（见下方网络说明）
  tokenizer_path: "/mnt/model/Qwen3-32B"   # 【必改】容器内路径，见上方挂载表
  llm_type: "vllm"    # 【必改】引擎类型: vllm / sglang / mindie
  enable_clear_hbm: true   # 每轮请求前重置 prefix cache（建议 true）
  stream: true    # 流式输出（建议 true，否则无法测 TTFT/TPOT）
  ignore_eos: true    # 忽略结束符（压测时建议 true，保证固定输出长度）
  timeout: 180    # 单请求超时（秒）
  verbose: false   # 详细日志：开启后保存每次请求/响应原文到 results/verbose/
  extra_info: ""  # 标记信息，用于区分不同服务拉起参数
```

> ⚠️ **server_url 网络说明（容器访问推理服务）**：
>
> 下方运行命令用了 `--network=host`，容器直接共享宿主机网络，所以：
>
> - 推理服务在**本机**上：直接填 `http://127.0.0.1:8000`。
> - 推理服务在**别的机器**上：填那台机器的 IP，如 `http://192.168.1.10:8000`。
>
> （如果你用的是 Docker Desktop / WSL2 而非原生 Linux，`--network=host` 行为不同，需改用 `--add-host=host.docker.internal:host-gateway` 并把 `server_url` 填 `http://host.docker.internal:8000`。）

## 6. 修改测试参数

打开代码目录里的 `suites/E2E/test_uc_performance.py`，找到 `perf_scenarios`：

```python
perf_scenarios = [
    # (mean_in, mean_out, max_req, concurrent, random_seed, hit_rate)
    (100, 10, 1, 1, 0, 50)
]
```

格式说明：

| 位置 | 字段          | 含义                                    | 示例       |
| ---- | ------------- | --------------------------------------- | ---------- |
| 1    | `mean_in`     | 输入 token 长度                         | `4096`     |
| 2    | `mean_out`    | 输出 token 长度                         | `1024`     |
| 3    | `max_req`     | 请求总数                                | `10`       |
| 4    | `concurrent`  | 并发数                                  | `1`        |
| 5    | `random_seed` | 随机种子（0 = 随机，固定值 = 可复现）   | `95273856` |
| 6    | `hit_rate`    | 命中率（0-100，0 = 不测命中，详见下文） | `80`       |

**示例：测一组 4096 输入 / 1024 输出 / 并发 1 / 命中率 80%**

```python
perf_scenarios = [
    (4096, 1024, 1, 1, 0, 80)
]
```

## 7. 后台运行测试

把下面命令里的两个路径换成你自己的，直接复制运行：

```bash
docker run -d --name=perf-test \
  --network=host \
  -v /mnt/d/Project/llmperf:/workspace \
  -v /mnt/model:/mnt/model \
  -w /workspace \
  llmperf:arm64-tf4 pytest suites/E2E/test_uc_performance.py -v
```

逐行解释：

| 命令片段                                   | 含义                                                       |
| ------------------------------------------ | ---------------------------------------------------------- |
| `-d`                                       | 后台运行                                                   |
| `--name=perf-test`                        | 容器名，方便后续查看日志                                   |
| `--network=host`                           | 容器直接用宿主机网络，`server_url` 可填 `127.0.0.1`        |
| `-v /mnt/d/Project/llmperf:/workspace`    | 宿主机代码 → 容器 `/workspace`（工作目录）                 |
| `-v /mnt/model:/mnt/model`                 | 宿主机模型 → 容器 `/mnt/model`（保持路径一致，配置不用改） |
| `-w /workspace`                            | 容器内工作目录                                             |
| `llmperf:arm64-tf4`                        | 镜像名（x86 + 非 GLM-5 改 `llmperf:x86_64-tf4`，GLM-5 改对应 `tf5`） |
| `pytest …`                                 | 要执行的命令                                               |

## 8. 查看运行日志

实时查看日志：

```bash
docker logs -f perf-test
```

保存日志到文件：

```bash
docker logs perf-test > output.log
```

## 9. 查看结果

结果写在容器内 `/workspace/results/`，因为代码目录是挂载的，所以**宿主机代码目录的 `results/` 里直接就能看到**：

| 文件                    | 内容                            |
| ----------------------- | ------------------------------- |
| `results/llmperf.jsonl` | 每条测试记录（JSON Lines 格式） |
| `results/llmperf.csv`   | 同内容 CSV，方便 Excel 打开     |

各字段含义见文末 [测试结果字段说明](#测试结果字段说明)。

## 10. 清理容器

测试完成后删除容器：

```bash
docker rm -f perf-test
```

---

# 测试结果字段说明

| 字段                            | 含义                              |
| ------------------------------- | --------------------------------- |
| `ttft_mean` / `ttft_p50/p90/p99` | 首 Token 延迟（秒）均值/分位数    |
| `tpot_mean` / `tpot_p50/p90/p99` | 每 Token 生成延迟（秒）均值/分位数 |
| `tbt_mean` / `tbt_p50/p90/p99`  | Token 间延迟（秒）均值/分位数     |
| `e2e_mean` / `e2e_p50/p90/p99`  | 端到端延迟（秒）均值/分位数       |
| `total_throughput`              | 整体吞吐（token/s）               |
| `incremental_throughput`        | 增量吞吐（token/s）              |
| `num_completed_requests`        | 完成请求数                        |
| `error_rate`                    | 错误率                            |

> 测试开始前会自动做一次服务健康检查（发一个最小请求），如果服务连不上会立即报错终止，不会浪费时间跑出一堆无效数据。

---

# 不同引擎命中策略说明

### vLLM

- 自动 prefix cache，只需设置 `hit_rate`
- 不需要重启服务
- 流程：预热注入（`mean_in × hit_rate%` 长度）→ 清缓存 → 正式压测

### SGLang

- 会做双重 prefill，不需要手动干预
- 流程：prefill_1（完整命中长度）→ prefill_2（短 prompt 触发）→ 清缓存 → 正式压测

### MindIE（纯盘命中场景）

MindIE 的纯盘命中需要特殊流程，否则测试结果不真实。原因：如果不重启服务，显存中的 KV Cache 会优先命中，而不是磁盘缓存。

**步骤 1：预热注入**

```python
perf_scenarios = [(4096, 1, 1, 1, 95273856, 80)]
```

运行测试，发送 80% 长度的请求，把 KV Cache 写入 UCM 磁盘。

**步骤 2：重启 MindIE 服务**

目的：清空显存 KV Cache，仅保留磁盘数据。

**步骤 3：正式压测**

```python
perf_scenarios = [(4096, 1024, 1, 1, 95273856, 100)]
```

运行：测到的性能是真实磁盘命中性能。**保证第 5 个参数（random_seed）前后一致**，才能获取与预埋时一致的输入数据。

---

# FAQ

### Q1：为什么有 tf4 和 tf5 两套镜像？我该用哪个？

GLM-5 系列模型的 tokenizer 需要用 `transformers >= 5.0`（推荐 `5.2.0`）才能正确加载；而其他模型（Qwen、Llama、DeepSeek 等）在 `transformers 5.x` 下可能不兼容，需要用 `4.57.6`。

所以提供了两套镜像：

| 镜像后缀 | transformers 版本 | 适用模型                         |
| -------- | ----------------- | -------------------------------- |
| `tf4`    | 4.57.6            | Qwen、Llama、DeepSeek 等非 GLM-5 |
| `tf5`    | 5.2.0             | GLM-5 系列                       |

- **Docker 用户**：根据模型选 `tf4` 或 `tf5` 镜像下载即可。
- **本地运行用户**：自己 `pip install` 对应版本的 transformers：
  ```bash
  # 非 GLM-5 模型
  pip install transformers==4.57.6
  # GLM-5 系列模型
  pip install transformers==5.2.0
  ```

### Q2：导入镜像后怎么确认 transformers 版本对不对？

加载镜像后跑一行检查命令：

```bash
docker run --rm llmperf:arm64-tf4 python -c "import transformers; print(transformers.__version__)"
```

输出 `4.57.6` 即正确（tf5 镜像应输出 `5.2.0`）。

### Q3：测试一开始就报"Service health check FAILED"怎么办？

这说明推理服务连不上。检查：

1. `config.yaml` 里的 `server_url` 是否正确（Docker 用户注意：容器内 `127.0.0.1` 是容器自己，访问宿主机服务要用 `host.docker.internal`）。
2. 推理服务是否已启动、端口是否开放。
3. `model` 名是否和服务端一致。

### Q4：结果里全是 NaN 是什么意思？

说明请求全部失败或没有成功完成。检查：
- 推理服务是否正常返回（看 verbose 日志）。
- tokenizer 路径是否正确（找不到 tokenizer 会报错）。
- 网络/防火墙是否放通了对应端口。
