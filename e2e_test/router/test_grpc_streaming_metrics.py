"""E2E test for gRPC streaming metrics (ITL, E2E latency, queue time).

Starts a mock SGLang gRPC server that streams token responses, launches
the SMG binary pointing to it, sends streaming requests, and verifies
that all per-request streaming metrics are recorded in the Prometheus endpoint.

This test does NOT require a GPU — it uses a lightweight mock gRPC server.

Usage:
    pytest e2e_test/router/test_grpc_streaming_metrics.py -v -s
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import io
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import grpc
import httpx
import pytest
from google.protobuf import struct_pb2

logger = logging.getLogger(__name__)

# ── Paths ──
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROTO_DIR = _REPO_ROOT / "crates" / "grpc_client" / "proto"


# ============================================================================
# Proto stub generation
# ============================================================================


def _generate_proto_stubs():
    """Generate Python gRPC stubs from sglang_scheduler.proto."""
    out_dir = tempfile.mkdtemp(prefix="smg_proto_")
    proto_files = ["sglang_scheduler.proto", "common.proto"]

    for proto_file in proto_files:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"--proto_path={_PROTO_DIR}",
                f"--python_out={out_dir}",
                f"--grpc_python_out={out_dir}",
                str(_PROTO_DIR / proto_file),
            ]
        )

    # Fix import paths in generated code
    sglang_pb2_path = Path(out_dir) / "sglang_scheduler_pb2.py"
    if sglang_pb2_path.exists():
        content = sglang_pb2_path.read_text()
        content = content.replace(
            "import common_pb2", "from . import common_pb2"
        )
        sglang_pb2_path.write_text(content)

    sglang_grpc_path = Path(out_dir) / "sglang_scheduler_pb2_grpc.py"
    if sglang_grpc_path.exists():
        content = sglang_grpc_path.read_text()
        content = content.replace(
            "import sglang_scheduler_pb2", "from . import sglang_scheduler_pb2"
        )
        content = content.replace(
            "import common_pb2", "from . import common_pb2"
        )
        sglang_grpc_path.write_text(content)

    # Also fix common_pb2_grpc.py if it has cross-imports
    common_grpc_path = Path(out_dir) / "common_pb2_grpc.py"
    if common_grpc_path.exists():
        content = common_grpc_path.read_text()
        content = content.replace(
            "import common_pb2", "from . import common_pb2"
        )
        common_grpc_path.write_text(content)

    # Create __init__.py so it's importable as a package
    (Path(out_dir) / "__init__.py").write_text("")

    return out_dir


# Generate stubs and import
try:
    _stub_dir = _generate_proto_stubs()
    sys.path.insert(0, str(Path(_stub_dir).parent))
    _pkg_name = Path(_stub_dir).name

    sglang_pb2 = importlib.import_module(f"{_pkg_name}.sglang_scheduler_pb2")
    sglang_pb2_grpc = importlib.import_module(
        f"{_pkg_name}.sglang_scheduler_pb2_grpc"
    )
    common_pb2 = importlib.import_module(f"{_pkg_name}.common_pb2")
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip(
        f"Cannot generate gRPC stubs (install grpcio-tools): {e}",
        allow_module_level=True,
    )


# ============================================================================
# Helpers
# ============================================================================


def get_open_port() -> int:
    """Get a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def find_smg_binary() -> str:
    """Find the SMG binary, checking common locations."""
    candidates = [
        _REPO_ROOT / "target" / "debug" / "smg",
        _REPO_ROOT / "target" / "release" / "smg",
    ]
    for p in candidates:
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    pytest.skip("SMG binary not found — run `cargo build --package smg --bin smg` first")


def wait_for_http(url: str, timeout: float = 30) -> None:
    """Wait for an HTTP endpoint to become reachable."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=2)
            if resp.status_code < 500:
                return
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}")


# ============================================================================
# Mock SGLang gRPC Server
# ============================================================================


class MockSglangScheduler(sglang_pb2_grpc.SglangSchedulerServicer):
    """Mock SGLang gRPC server that returns canned streaming responses."""

    NUM_CHUNKS = 5
    TOKENS_PER_CHUNK = 3
    CHUNK_DELAY_SECS = 0.02  # 20ms between chunks for measurable ITL

    def Generate(self, request, context):
        """Stream back NUM_CHUNKS chunks, then a Complete message."""
        prompt_tokens = len(request.tokenized.input_ids) or 10
        total_completion = 0

        for i in range(self.NUM_CHUNKS):
            total_completion += self.TOKENS_PER_CHUNK
            chunk = sglang_pb2.GenerateStreamChunk(
                token_ids=list(
                    range(
                        100 + i * self.TOKENS_PER_CHUNK,
                        100 + (i + 1) * self.TOKENS_PER_CHUNK,
                    )
                ),
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion,
                cached_tokens=0,
                index=0,
            )
            yield sglang_pb2.GenerateResponse(
                request_id=request.request_id,
                chunk=chunk,
            )
            time.sleep(self.CHUNK_DELAY_SECS)

        # Send Complete
        complete = sglang_pb2.GenerateComplete(
            output_ids=list(range(100, 100 + total_completion)),
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=total_completion,
            cached_tokens=0,
            index=0,
        )
        yield sglang_pb2.GenerateResponse(
            request_id=request.request_id,
            complete=complete,
        )

    def HealthCheck(self, request, context):
        return sglang_pb2.HealthCheckResponse(healthy=True, message="ok")

    def GetModelInfo(self, request, context):
        return sglang_pb2.GetModelInfoResponse(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            tokenizer_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            is_generation=True,
            served_model_name="mock-model",
            max_context_length=4096,
            vocab_size=32000,
            supports_vision=False,
            model_type="llama",
        )

    def GetServerInfo(self, request, context):
        return sglang_pb2.GetServerInfoResponse(
            server_args=struct_pb2.Struct(),
            scheduler_info=struct_pb2.Struct(),
            sglang_version="0.0.0-mock",
            server_type="grpc",
        )

    def Abort(self, request, context):
        return sglang_pb2.AbortResponse(success=True, message="ok")

    def GetLoads(self, request, context):
        return sglang_pb2.GetLoadsResponse(
            timestamp="2025-01-01T00:00:00Z",
            version="0.0.0-mock",
            dp_rank_count=1,
        )

    def Embed(self, request, context):
        return sglang_pb2.EmbedResponse(
            request_id=request.request_id,
            complete=sglang_pb2.EmbedComplete(
                embedding=[0.0] * 128,
                prompt_tokens=10,
            ),
        )

    def GetTokenizer(self, request, context):
        """Serve a minimal tokenizer zip bundle.

        The SMG gateway expects a zip archive containing tokenizer.json.
        We serve a minimal HuggingFace tokenizer that can encode basic ASCII.
        """
        # Minimal tokenizer.json that tokenizers crate can load
        tokenizer_json = """{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {"hello": 0, "world": 1, " ": 2, "count": 3, "to": 4, "five": 5, "Count": 6},
    "merges": []
  }
}"""
        # Create a zip archive in memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("tokenizer.json", tokenizer_json)
        zip_bytes = buf.getvalue()
        sha256 = hashlib.sha256(zip_bytes).hexdigest()

        # Stream it as a single chunk with the sha256 set
        yield common_pb2.GetTokenizerChunk(data=zip_bytes, sha256=sha256)

    def SubscribeKvEvents(self, request, context):
        # Return empty stream (no events)
        return
        yield  # noqa: unreachable — makes this a generator


def start_mock_grpc_server(port: int) -> grpc.Server:
    """Start a mock SGLang gRPC server on the given port."""
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=4))
    sglang_pb2_grpc.add_SglangSchedulerServicer_to_server(
        MockSglangScheduler(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("Mock gRPC server started on port %d", port)
    return server


# ============================================================================
# Metrics Parsing Helpers
# ============================================================================


def fetch_metrics(metrics_url: str) -> str:
    """Fetch Prometheus metrics text from the gateway."""
    resp = httpx.get(f"{metrics_url}/metrics", timeout=10)
    resp.raise_for_status()
    return resp.text


def metric_exists(metrics_text: str, metric_name: str) -> bool:
    """Check if a metric name appears in the Prometheus exposition text."""
    return any(
        line.startswith(f"{metric_name}_") or line.startswith(f"{metric_name}{{")
        for line in metrics_text.splitlines()
        if not line.startswith("#")
    )


def get_metric_sum(metrics_text: str, metric_name: str) -> float | None:
    """Extract the _sum value for a histogram metric."""
    target = f"{metric_name}_sum"
    for line in metrics_text.splitlines():
        if line.startswith(target):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[-1])
    return None


def get_metric_count(metrics_text: str, metric_name: str) -> float | None:
    """Extract the _count value for a histogram metric."""
    target = f"{metric_name}_count"
    for line in metrics_text.splitlines():
        if line.startswith(target):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[-1])
    return None


# ============================================================================
# Test
# ============================================================================


@pytest.mark.skipif(
    not (_PROTO_DIR / "sglang_scheduler.proto").exists(),
    reason="Proto files not found",
)
class TestGrpcStreamingMetrics:
    """Test that ITL, E2E latency, and queue time metrics are recorded for gRPC streaming."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_grpc_backend(self, request):
        """Start mock gRPC server + SMG gateway binary."""
        smg_bin = find_smg_binary()
        grpc_port = get_open_port()
        gateway_port = get_open_port()
        prometheus_port = get_open_port()

        # Start mock gRPC server
        grpc_server = start_mock_grpc_server(grpc_port)

        # Launch SMG binary
        worker_url = f"grpc://127.0.0.1:{grpc_port}"
        cmd = [
            smg_bin,
            "launch",
            "--host", "127.0.0.1",
            "--port", str(gateway_port),
            "--prometheus-port", str(prometheus_port),
            "--prometheus-host", "127.0.0.1",
            "--worker-urls", worker_url,
            "--log-level", "info",
        ]
        logger.info("Launching SMG: %s", " ".join(cmd))

        smg_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        base_url = f"http://127.0.0.1:{gateway_port}"
        metrics_url = f"http://127.0.0.1:{prometheus_port}"

        try:
            # Wait for gateway to be ready
            wait_for_http(f"{base_url}/health", timeout=30)
            # Wait for worker to be discovered
            wait_for_http(f"{base_url}/v1/workers", timeout=30)
            logger.info("SMG gateway ready at %s", base_url)
        except TimeoutError:
            # Dump logs on failure
            smg_process.terminate()
            stdout, _ = smg_process.communicate(timeout=5)
            if stdout:
                logger.error("SMG logs:\n%s", stdout.decode(errors="replace")[-2000:])
            grpc_server.stop(grace=1)
            raise

        # Store on class
        request.cls.base_url = base_url
        request.cls.metrics_url = metrics_url
        request.cls.smg_process = smg_process
        request.cls.grpc_server = grpc_server

        yield

        # Cleanup
        os.killpg(os.getpgid(smg_process.pid), signal.SIGTERM)
        smg_process.wait(timeout=10)
        grpc_server.stop(grace=1)

    def _send_streaming_request(self, n_requests: int = 1):
        """Send streaming generate requests through the gateway.

        Uses the /generate endpoint which doesn't require chat template rendering.
        """
        for _ in range(n_requests):
            resp = httpx.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"max_new_tokens": 10, "temperature": 0.0},
                    "stream": True,
                    "model": "mock-model",
                },
                timeout=30,
            )
            if resp.status_code >= 400:
                raise AssertionError(
                    f"Generate request failed: {resp.status_code} {resp.text[:500]}"
                )
            chunks = []
            for line in resp.text.split("\n"):
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunks.append(data)
            assert len(chunks) > 0, "Expected at least one streaming chunk"

    def test_streaming_metrics_recorded(self):
        """Verify ITL, E2E latency, and queue time metrics appear after streaming requests."""
        self._send_streaming_request(n_requests=3)
        time.sleep(0.5)

        metrics_text = fetch_metrics(self.metrics_url)

        # ── ITL (inter-token latency) ──
        assert metric_exists(metrics_text, "smg_router_itl_seconds"), (
            "smg_router_itl_seconds not found in Prometheus metrics"
        )
        itl_count = get_metric_count(metrics_text, "smg_router_itl_seconds")
        assert itl_count is not None and itl_count > 0, (
            f"smg_router_itl_seconds_count should be > 0, got {itl_count}"
        )
        itl_sum = get_metric_sum(metrics_text, "smg_router_itl_seconds")
        assert itl_sum is not None and itl_sum > 0, (
            f"smg_router_itl_seconds_sum should be > 0, got {itl_sum}"
        )

        # ── E2E request latency ──
        assert metric_exists(
            metrics_text, "smg_router_e2e_request_latency_seconds"
        ), "smg_router_e2e_request_latency_seconds not found in Prometheus metrics"
        e2e_count = get_metric_count(
            metrics_text, "smg_router_e2e_request_latency_seconds"
        )
        assert e2e_count is not None and e2e_count > 0, (
            f"smg_router_e2e_request_latency_seconds_count should be > 0, got {e2e_count}"
        )

        # ── Queue time ──
        assert metric_exists(metrics_text, "smg_router_queue_time_seconds"), (
            "smg_router_queue_time_seconds not found in Prometheus metrics"
        )
        queue_count = get_metric_count(
            metrics_text, "smg_router_queue_time_seconds"
        )
        assert queue_count is not None and queue_count > 0, (
            f"smg_router_queue_time_seconds_count should be > 0, got {queue_count}"
        )

        # ── Existing metrics still present ──
        assert metric_exists(metrics_text, "smg_router_ttft_seconds"), (
            "smg_router_ttft_seconds should still be present"
        )
        assert metric_exists(metrics_text, "smg_router_tpot_seconds"), (
            "smg_router_tpot_seconds should still be present"
        )
        assert metric_exists(
            metrics_text, "smg_router_generation_duration_seconds"
        ), "smg_router_generation_duration_seconds should still be present"

    def test_itl_values_plausible(self):
        """Verify ITL values are plausible given the mock server's 20ms chunk delay."""
        self._send_streaming_request(n_requests=2)
        time.sleep(0.5)

        metrics_text = fetch_metrics(self.metrics_url)

        itl_sum = get_metric_sum(metrics_text, "smg_router_itl_seconds")
        itl_count = get_metric_count(metrics_text, "smg_router_itl_seconds")

        assert itl_sum is not None and itl_count is not None and itl_count > 0

        avg_itl = itl_sum / itl_count
        # Mock sends 3 tokens per chunk with 20ms delay between chunks
        # So per-token ITL ~ 20ms / 3 ~ 6.7ms
        # Allow generous range: 1ms to 200ms (network + scheduling jitter)
        assert 0.001 < avg_itl < 0.200, (
            f"Average ITL {avg_itl:.4f}s is outside plausible range [1ms, 200ms]"
        )

    def test_e2e_greater_than_generation_duration(self):
        """E2E latency should be >= generation_duration (includes pipeline overhead)."""
        self._send_streaming_request(n_requests=2)
        time.sleep(0.5)

        metrics_text = fetch_metrics(self.metrics_url)

        e2e_sum = get_metric_sum(
            metrics_text, "smg_router_e2e_request_latency_seconds"
        )
        gen_sum = get_metric_sum(
            metrics_text, "smg_router_generation_duration_seconds"
        )

        assert e2e_sum is not None and gen_sum is not None
        assert e2e_sum >= gen_sum, (
            f"E2E latency sum ({e2e_sum:.4f}s) should be >= generation duration sum ({gen_sum:.4f}s)"
        )

    def test_queue_time_non_negative(self):
        """Queue time should be non-negative."""
        self._send_streaming_request(n_requests=1)
        time.sleep(0.5)

        metrics_text = fetch_metrics(self.metrics_url)

        queue_sum = get_metric_sum(metrics_text, "smg_router_queue_time_seconds")
        assert queue_sum is not None and queue_sum >= 0, (
            f"Queue time sum should be >= 0, got {queue_sum}"
        )
