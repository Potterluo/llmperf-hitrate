import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
from common.config_utils import config_utils
from common.llmperf.utils import common_metrics
from common.llmperf.utils.models import RequestConfig

logger = logging.getLogger(__name__)

# Sentinel error codes for failures that do not carry an HTTP status code.
ERR_CONNECTION = "CONNECTION_ERROR"
ERR_TIMEOUT = "TIMEOUT"
ERR_PARSE = "PARSE_ERROR"
ERR_STREAM = "STREAM_ERROR"


def _get_config(name: str, default):
    """Read config lazily so changes after module import are respected."""
    return config_utils.get_nested_config(f"llm_connection.{name}", default)


def check_service_health(
    server_url: str, model: str, timeout: int = 30
) -> Tuple[bool, str]:
    """Probe the inference service with a minimal non-streaming request.

    Returns ``(ok, message)``. Use this before a benchmark run so that an
    unreachable / misconfigured service fails fast instead of producing a
    batch of meaningless zero/nan metrics.
    """
    url = server_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/chat/completions"

    key = os.environ.get("OPENAI_API_KEY", "secret_abcdefg")
    headers = {"Authorization": f"Bearer {key}"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
        "ignore_eos": True,
    }

    try:
        resp = requests.post(url, json=body, timeout=timeout, headers=headers)
    except requests.Timeout:
        return False, f"service health check timed out ({timeout}s): {url}"
    except requests.ConnectionError as e:
        return False, f"cannot connect to service {url}: {e}"
    except requests.RequestException as e:
        return False, f"service health check request error: {e}"

    if resp.status_code == 200:
        return True, "service is healthy"
    return (
        False,
        f"service returned HTTP {resp.status_code}: {resp.text[:200]}",
    )


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of raising on a zero denominator."""
    return numerator / denominator if denominator else 0.0


class OpenAIChatCompletionsClient:
    """
    Sends HTTP requests to an OpenAI-compatible chat/completions endpoint,
    consumes the token stream, and measures latency metrics (TTFT, TPOT,
    inter-token latency, end-to-end latency, throughput).
    """

    def llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[Dict[str, Any], str, RequestConfig]:
        stream = _get_config("stream", True)
        ignore_eos = _get_config("ignore_eos", True)
        timeout = _get_config("timeout", 180)
        verbose = _get_config("verbose", False)

        prompt, prompt_len = request_config.prompt

        message = [
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": stream,
            "ignore_eos": ignore_eos,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        time_to_next_token = []
        tokens_received = 0
        ttft = 0.0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0.0
        total_request_time = 0.0
        first_token_seen = False
        request_ok = False

        metrics: Dict[str, Any] = {}
        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = start_time

        address = request_config.openai_api_base
        if not address:
            raise ValueError("the environment variable OPENAI_API_BASE must be set.")
        key = os.environ.get("OPENAI_API_KEY", "secret_abcdefg")
        if not key:
            raise ValueError("the environment variable OPENAI_API_KEY must be set.")
        headers = {"Authorization": f"Bearer {key}"}
        if not address.endswith("/"):
            address = address + "/"
        address += "chat/completions"

        if verbose:
            logger.info(
                "[verbose] request → %s | model=%s stream=%s max_tokens=%s "
                "input_tokens=%s",
                address,
                model,
                stream,
                body.get("max_tokens"),
                prompt_len,
            )

        try:
            with requests.post(
                address,
                json=body,
                stream=stream,
                timeout=timeout,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()

                for chunk in response.iter_lines(chunk_size=None):
                    if not chunk:
                        continue
                    stem = b"data: "
                    if chunk.startswith(stem):
                        chunk = chunk[len(stem):]
                    # Data might already be bytes or str
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8", errors="ignore")
                    if chunk.strip() == "[DONE]":
                        continue
                    tokens_received += 1
                    data = json.loads(chunk)
                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"].get("code") or ERR_PARSE
                        raise RuntimeError(error_msg)
                    delta = data["choices"][0]["delta"]
                    content = delta.get("content") or delta.get(
                        "reasoning_content", ""
                    )
                    if content:
                        if not first_token_seen:
                            # First real content token → record TTFT.
                            ttft = time.monotonic() - start_time
                            first_token_seen = True
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += content

            total_request_time = time.monotonic() - start_time
            output_throughput = _safe_div(tokens_received, total_request_time)
            request_ok = True

        except requests.Timeout as e:
            error_response_code = error_response_code or ERR_TIMEOUT
            error_msg = error_msg or str(e)
            logger.warning("Request timed out (code=%s)", error_response_code)
        except requests.ConnectionError as e:
            error_response_code = error_response_code or ERR_CONNECTION
            error_msg = error_msg or str(e)
            logger.warning("Connection failed: %s", e)
        except requests.HTTPError as e:
            error_response_code = error_response_code or e.response.status_code
            error_msg = error_msg or str(e)
            logger.warning("HTTP error (code=%s)", error_response_code)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            error_response_code = error_response_code or ERR_PARSE
            error_msg = error_msg or str(e)
            logger.warning("Failed to parse stream response: %s", e)
        except Exception as e:  # noqa: BLE001 - last resort, never crash the run
            error_response_code = error_response_code or ERR_STREAM
            error_msg = error_msg or str(e)
            logger.warning("Request failed: %s", e)

        # If the request never completed cleanly, the partially-measured
        # numbers are not trustworthy — record the failure and zero the
        # derived metrics so they cannot poison the aggregate statistics
        # (and cannot trigger a divide-by-zero downstream).
        if not request_ok:
            total_request_time = 0.0
            ttft = 0.0
            output_throughput = 0.0
            time_to_next_token.clear()
            tokens_received = 0
            generated_text = ""

        metrics[common_metrics.ERROR_MSG] = error_msg
        metrics[common_metrics.ERROR_CODE] = error_response_code

        # Inter-token latency (TBT): mean of per-gap intervals.
        inter_token_lat = (
            sum(time_to_next_token) / len(time_to_next_token)
            if time_to_next_token
            else 0.0
        )
        # TPOT (Time Per Output Token): generation time after the first token,
        # averaged over the remaining output tokens.
        decode_time = max(total_request_time - ttft, 0.0)
        tpot = _safe_div(decode_time, max(tokens_received - 1, 0))

        metrics[common_metrics.INTER_TOKEN_LAT] = inter_token_lat
        metrics[common_metrics.TPOT] = tpot
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        if verbose:
            logger.info(
                "[verbose] response ← ok=%s ttft=%.4fs tpot=%.6fs tbt_mean=%.6fs "
                "e2e=%.4fs out_tokens=%d err=%s",
                request_ok,
                ttft,
                tpot,
                inter_token_lat,
                total_request_time,
                tokens_received,
                error_response_code,
            )
            self._dump_exchange(model, body, generated_text, metrics)

        return metrics, generated_text, request_config

    @staticmethod
    def _dump_exchange(
        model: str, body: Dict[str, Any], generated_text: str, metrics: Dict[str, Any]
    ) -> None:
        """Persist the raw request/response exchange to a log file for debugging."""
        try:
            backup_dir = Path(
                config_utils.get_nested_config("database.backup", "results/")
            ).resolve()
            log_dir = backup_dir / "verbose"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"exchange_{int(time.time() * 1000)}.json"
            payload = {
                "model": model,
                "request_body": body,
                "response_preview": generated_text[:512],
                "response_length": len(generated_text),
                "metrics": metrics,
            }
            with log_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:  # noqa: BLE001 - logging must never break the run
            logger.debug("Failed to dump verbose exchange: %s", e)
