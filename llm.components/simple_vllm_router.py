from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import time


# ----------------------------------------------------------------------
# Request / replica models
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    request_id: str
    model: str

    # Optional requirements.
    lora_id: Optional[str] = None
    required_context_len: int = 0

    # Optional prefix identifier. In a real system this might be derived
    # from prompt blocks or a prefix-cache key.
    prefix_hash: Optional[str] = None


@dataclass
class ReplicaMetrics:
    # Number of requests currently executing inside vLLM.
    running_requests: int

    # Requests waiting for admission/scheduling.
    waiting_requests: int

    # Number of tokens currently scheduled by the engine.
    scheduled_tokens: int

    # KV-cache utilization in [0.0, 1.0].
    kv_cache_usage: float

    # GPU utilization in [0.0, 1.0].
    gpu_utilization: float

    # Unix timestamp when these metrics were collected.
    timestamp: float


@dataclass
class Replica:
    replica_id: str
    endpoint: str

    model: str
    max_context_len: int

    healthy: bool
    ready: bool

    metrics: ReplicaMetrics

    # Optional LoRAs already resident on this replica.
    loaded_loras: set[str]

    # Prefix hashes known to be cached locally.
    cached_prefixes: set[str]


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------

class NoBackendAvailable(Exception):
    pass


class VLLMRouter:
    def __init__(
        self,
        *,
        max_metric_age_seconds: float = 2.0,
        max_kv_usage: float = 0.95,
        max_waiting_requests: int = 100,
    ):
        self.max_metric_age_seconds = max_metric_age_seconds
        self.max_kv_usage = max_kv_usage
        self.max_waiting_requests = max_waiting_requests

    def choose_backend(
        self,
        request: Request,
        replicas: list[Replica],
        now: Optional[float] = None,
    ) -> Replica:

        if now is None:
            now = time.time()

        candidates: list[Replica] = []

        # --------------------------------------------------------------
        # Phase 1: hard filtering
        # --------------------------------------------------------------

        for replica in replicas:

            # Health/readiness.
            if not replica.healthy or not replica.ready:
                continue

            # Model compatibility.
            if replica.model != request.model:
                continue

            # Context compatibility.
            if request.required_context_len > replica.max_context_len:
                continue

            # LoRA compatibility.
            #
            # This example requires the LoRA to already be loaded.
            # Another design could allow loading it dynamically and
            # incorporate loading cost into the score.
            if (
                request.lora_id is not None
                and request.lora_id not in replica.loaded_loras
            ):
                continue

            metrics = replica.metrics

            # ----------------------------------------------------------
            # Reject stale metrics.
            # ----------------------------------------------------------

            metric_age = now - metrics.timestamp

            if metric_age < 0:
                # Clock skew or invalid metric.
                continue

            if metric_age > self.max_metric_age_seconds:
                continue

            # ----------------------------------------------------------
            # Avoid obviously overloaded replicas.
            # ----------------------------------------------------------

            if metrics.kv_cache_usage >= self.max_kv_usage:
                continue

            if metrics.waiting_requests >= self.max_waiting_requests:
                continue

            candidates.append(replica)

        if not candidates:
            raise NoBackendAvailable(
                f"No compatible backend for model={request.model}"
            )

        # --------------------------------------------------------------
        # Phase 2: score candidates
        # --------------------------------------------------------------

        return min(
            candidates,
            key=lambda replica: self._score(request, replica, now),
        )

    def _score(
        self,
        request: Request,
        replica: Replica,
        now: float,
    ) -> float:

        m = replica.metrics

        # --------------------------------------------------------------
        # Queue pressure
        # --------------------------------------------------------------

        queue_score = (
            2.0 * m.waiting_requests
            + 0.5 * m.running_requests
        )

        # --------------------------------------------------------------
        # Token workload
        #
        # Request count alone is misleading for LLMs. A replica with
        # two very large requests may be more loaded than one with ten
        # short requests.
        # --------------------------------------------------------------

        token_score = m.scheduled_tokens / 1024.0

        # --------------------------------------------------------------
        # KV pressure
        #
        # Make this nonlinear so that a nearly-full KV cache becomes
        # increasingly unattractive.
        #
        # 50% -> 1
        # 80% -> 4
        # 90% -> 9
        # --------------------------------------------------------------

        remaining_kv = max(0.01, 1.0 - m.kv_cache_usage)

        kv_score = (
            m.kv_cache_usage / remaining_kv
        )

        # --------------------------------------------------------------
        # GPU load
        #
        # GPU utilization should not be the primary signal because
        # decode may be memory-bandwidth bound.
        # --------------------------------------------------------------

        gpu_score = m.gpu_utilization

        # --------------------------------------------------------------
        # Metric age
        #
        # We already reject metrics that are too old, but among valid
        # replicas prefer fresher information.
        # --------------------------------------------------------------

        metric_age = max(0.0, now - m.timestamp)

        freshness_penalty = (
            metric_age / self.max_metric_age_seconds
        )

        # --------------------------------------------------------------
        # Prefix-cache affinity
        #
        # Cache affinity is a preference, not a hard requirement.
        # Don't send work to an overloaded replica just because it
        # happens to contain the prefix.
        # --------------------------------------------------------------

        prefix_bonus = 0.0

        if (
            request.prefix_hash is not None
            and request.prefix_hash in replica.cached_prefixes
        ):
            prefix_bonus = 5.0

        # --------------------------------------------------------------
        # Combined score
        #
        # Lower = better.
        #
        # These weights should eventually be learned/tuned from
        # production measurements.
        # --------------------------------------------------------------

        score = (
            1.0 * queue_score
            + 1.5 * token_score
            + 3.0 * kv_score
            + 0.5 * gpu_score
            + 2.0 * freshness_penalty
            - 1.0 * prefix_bonus
        )

        return score


# ----------------------------------------------------------------------
# Example
# ----------------------------------------------------------------------

if __name__ == "__main__":

    now = time.time()

    replicas = [
        Replica(
            replica_id="vllm-1",
            endpoint="http://vllm-1:8000",
            model="llama-3-8b",
            max_context_len=8192,
            healthy=True,
            ready=True,
            metrics=ReplicaMetrics(
                running_requests=20,
                waiting_requests=30,
                scheduled_tokens=20_000,
                kv_cache_usage=0.90,
                gpu_utilization=0.90,
                timestamp=now - 0.1,
            ),
            loaded_loras=set(),
            cached_prefixes={"prefix-A"},
        ),

        Replica(
            replica_id="vllm-2",
            endpoint="http://vllm-2:8000",
            model="llama-3-8b",
            max_context_len=8192,
            healthy=True,
            ready=True,
            metrics=ReplicaMetrics(
                running_requests=8,
                waiting_requests=2,
                scheduled_tokens=4_000,
                kv_cache_usage=0.45,
                gpu_utilization=0.65,
                timestamp=now - 0.2,
            ),
            loaded_loras=set(),
            cached_prefixes=set(),
        ),

        Replica(
            replica_id="vllm-3",
            endpoint="http://vllm-3:8000",
            model="llama-3-8b",
            max_context_len=8192,
            healthy=True,
            ready=True,
            metrics=ReplicaMetrics(
                running_requests=3,
                waiting_requests=0,
                scheduled_tokens=1_000,
                kv_cache_usage=0.30,
                gpu_utilization=0.40,

                # Deliberately stale.
                timestamp=now - 10.0,
            ),
            loaded_loras=set(),
            cached_prefixes={"prefix-A"},
        ),
    ]

    request = Request(
        request_id="req-123",
        model="llama-3-8b",
        required_context_len=4096,
        prefix_hash="prefix-A",
    )

    router = VLLMRouter(
        max_metric_age_seconds=2.0,
        max_kv_usage=0.95,
        max_waiting_requests=100,
    )

    try:
        selected = router.choose_backend(
            request=request,
            replicas=replicas,
            now=now,
        )

        print(
            f"Selected {selected.replica_id}: "
            f"{selected.endpoint}"
        )

    except NoBackendAvailable as e:
        print(f"503 Service Unavailable: {e}")


#####################


