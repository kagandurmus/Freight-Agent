"""The asynchronous worker pool that drains the triage outbox.

Structure:

    asyncio.TaskGroup
      |- worker-0  ---+
      |- worker-1     |   claim one job (lease) -> triage -> dispatch -> complete
      |- worker-2     |   on failure: backoff and requeue, or dead-letter
      +- worker-3  ---+

Design notes that matter in production rather than in a demo:

* **No busy polling.** Workers park on an `asyncio.Event` that the API sets the moment an
  event is accepted; the poll interval is only a safety net. The event is cleared *before*
  the claim attempt, so a job enqueued between the claim and the sleep cannot be missed.
* **Leases, not locks.** A worker that dies mid-triage holds nothing: its lease expires and
  another worker reclaims the job. That is why `stop()` can cancel in-flight work safely.
* **Backoff with jitter.** Retries spread out instead of synchronising into a thundering herd
  against a model provider that is already struggling.
* **Idle workers do maintenance.** A timed-out wait is used to sweep notifications orphaned by
  a crash, so recovery costs no extra timer.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from dataclasses import dataclass, field

from config import Settings
from store import Job, JobStatus, Store, utcnow
from triage import NotificationDispatcher, TriageEngineError, TriageService

__all__ = ["TriageWorker", "WorkerStats", "compute_backoff"]

logger = logging.getLogger(__name__)


def compute_backoff(
    attempts: int,
    *,
    base_seconds: float,
    max_seconds: float,
    jitter_ratio: float = 0.25,
    random_value: float | None = None,
) -> float:
    """Exponential backoff with additive jitter, capped.

    `attempts` is 1 for the first failure. Jitter is injected rather than drawn internally so
    the behaviour is testable.
    """
    exponent = max(0, attempts - 1)
    delay = min(base_seconds * (2**exponent), max_seconds)
    draw = random.random() if random_value is None else random_value
    return min(delay + delay * jitter_ratio * draw, max_seconds)


@dataclass
class WorkerStats:
    """Counters exposed on /healthz so operators can see the pipeline breathe."""

    claimed: int = 0
    completed: int = 0
    retried: int = 0
    dead_lettered: int = 0
    notifications_sent: int = 0
    recovered_leases: int = 0
    last_error: str | None = None
    per_worker: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "claimed": self.claimed,
            "completed": self.completed,
            "retried": self.retried,
            "dead_lettered": self.dead_lettered,
            "notifications_sent": self.notifications_sent,
            "recovered_leases": self.recovered_leases,
            "last_error": self.last_error,
            "per_worker": dict(self.per_worker),
        }


class TriageWorker:
    """Owns the worker tasks, their lifecycle and their counters."""

    def __init__(
        self,
        *,
        store: Store,
        service: TriageService,
        dispatcher: NotificationDispatcher,
        settings: Settings,
    ):
        self.store = store
        self.service = service
        self.dispatcher = dispatcher
        self.settings = settings
        self.stats = WorkerStats()
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._supervisor: asyncio.Task[None] | None = None
        #: Guards the crash-recovery sweep so two idle workers cannot deliver the same
        #: orphaned notification. Cross-process exactly-once delivery needs a claim state on
        #: the notification row itself, which belongs to the notification service (Step 5).
        self._sweep_lock = asyncio.Lock()

    # -- lifecycle --------------------------------------------------------------------

    async def start(self) -> None:
        """Reclaim abandoned jobs, then run the pool in a supervised TaskGroup."""
        self.stats.recovered_leases = await self.store.release_expired_leases()
        if self.stats.recovered_leases:
            logger.warning(
                "reclaimed %s job(s) whose worker died mid-triage",
                self.stats.recovered_leases,
            )
        self._stop.clear()
        self._supervisor = asyncio.create_task(self._supervise(), name="triage-supervisor")
        logger.info("triage worker pool started with %s workers", self.settings.worker_count)

    async def _supervise(self) -> None:
        async with asyncio.TaskGroup() as group:
            for index in range(self.settings.worker_count):
                group.create_task(self._loop(index), name=f"triage-worker-{index}")

    async def stop(self) -> None:
        """Stop claiming, let in-flight work drain, then cancel whatever remains."""
        self._stop.set()
        self._wake.set()  # release anyone parked on the event
        supervisor = self._supervisor
        if supervisor is None:
            return
        try:
            async with asyncio.timeout(self.settings.shutdown_grace_seconds):
                await supervisor
        except TimeoutError:
            logger.warning(
                "workers did not drain within %ss; cancelling (leases will expire)",
                self.settings.shutdown_grace_seconds,
            )
            supervisor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await supervisor
        self._supervisor = None
        logger.info("triage worker pool stopped: %s", self.stats.as_dict())

    def notify(self) -> None:
        """Called by the API after a durable ingest: work is waiting."""
        self._wake.set()

    # -- loop -------------------------------------------------------------------------

    async def _loop(self, index: int) -> None:
        name = f"worker-{index}"
        while not self._stop.is_set():
            # Clear before claiming: anything enqueued after this point sets the event and
            # the wait below returns immediately instead of sleeping through the work.
            self._wake.clear()
            try:
                jobs = await self.store.claim_jobs(
                    limit=1, lease_seconds=self.settings.job_lease_seconds
                )
            except Exception as exc:  # a database hiccup must not kill the worker
                self.stats.last_error = f"claim failed: {exc}"
                logger.exception("%s could not claim a job", name)
                await self._sleep(1.0)
                continue

            if jobs:
                self.stats.claimed += 1
                self.stats.per_worker[name] = self.stats.per_worker.get(name, 0) + 1
                await self._handle(jobs[0])
                continue

            if await self._wait_for_work():
                await self._maintain()
        logger.debug("%s observed shutdown", name)

    async def _wait_for_work(self) -> bool:
        """Park until notified or the poll interval elapses. True when the wait timed out."""
        try:
            async with asyncio.timeout(self.settings.worker_poll_interval_seconds):
                await self._wake.wait()
        except TimeoutError:
            return True
        return False

    async def _sleep(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await self._stop.wait()

    # -- work -------------------------------------------------------------------------

    async def _handle(self, job: Job) -> None:
        try:
            result = await self.service.triage_event(job.event_id, job=job)
        except TriageEngineError as exc:
            await self._reschedule(job, exc, retryable=True)
            return
        except asyncio.CancelledError:
            # Shutting down mid-triage: leave the lease to expire so the job is reclaimed.
            logger.info("job %s cancelled during shutdown; lease will expire", job.job_id)
            raise
        except Exception as exc:
            await self._reschedule(job, exc, retryable=False)
            return

        self.stats.completed += 1
        if result is None:
            return
        if result.action.value == "AUTO_EMAIL":
            sent = await self.dispatcher.dispatch_for_decision(result.decision_id)
            self.stats.notifications_sent += sent
            if sent:
                logger.info(
                    "decision %s notified %s recipient(s)", result.decision_id, sent
                )

    async def _reschedule(self, job: Job, exc: BaseException, *, retryable: bool) -> None:
        delay = compute_backoff(
            job.attempts,
            base_seconds=self.settings.job_backoff_base_seconds,
            max_seconds=self.settings.job_backoff_max_seconds,
        )
        message = f"{type(exc).__name__}: {exc}" if not isinstance(exc, Exception) else str(exc)
        self.stats.last_error = f"{job.event_id}: {message}"[:500]
        if not retryable:
            # An unexpected error is treated as exhausting the budget immediately: retrying a
            # bug just delays the operator seeing it.
            job = Job(job.job_id, job.event_id, job.max_attempts, job.max_attempts, job.kind)
        status = await self.store.fail_job(job, error=message, retry_in_seconds=delay)
        if status is JobStatus.DEAD:
            self.stats.dead_lettered += 1
            logger.error("job %s dead-lettered after %s attempts: %s", job.job_id, job.attempts, message)
        else:
            self.stats.retried += 1
            logger.warning(
                "job %s failed (attempt %s/%s), retrying in %.1fs: %s",
                job.job_id,
                job.attempts,
                job.max_attempts,
                delay,
                message,
            )

    async def _maintain(self) -> None:
        """Runs when a worker has nothing to do: recover notifications orphaned by a crash."""
        async with self._sweep_lock:
            try:
                delivered = await self.dispatcher.sweep()
            except Exception:
                logger.exception("notification sweep failed")
                return
        if delivered:
            self.stats.notifications_sent += delivered
            logger.info("sweep delivered %s orphaned notification(s)", delivered)
