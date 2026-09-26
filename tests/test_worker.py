"""Worker pool behaviour: retries, dead-lettering and crash recovery."""

from __future__ import annotations

import asyncio
import time

import pytest

from conftest import StubEngine, make_event, make_sla
from schemas import TriagePolicy
from store import NotificationStatus
from triage import NotificationDispatcher, TriageEngineError, TriageService
from worker import TriageWorker, compute_backoff


async def wait_for_decision(store, event_id: str, *, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        decision = await store.get_decision_for_event(event_id)
        if decision is not None:
            return decision
        await asyncio.sleep(0.02)
    return None


def build_worker(store, settings, engine, *, sent: list | None = None) -> TriageWorker:
    async def transport(notification):
        if sent is not None:
            sent.append(notification)

    dispatcher = NotificationDispatcher(store, transport=transport)
    service = TriageService(store, engine, TriagePolicy())
    return TriageWorker(store=store, service=service, dispatcher=dispatcher, settings=settings)


def test_backoff_grows_exponentially_and_is_capped():
    small = compute_backoff(1, base_seconds=1.0, max_seconds=60.0, random_value=0.0)
    second = compute_backoff(2, base_seconds=1.0, max_seconds=60.0, random_value=0.0)
    tenth = compute_backoff(10, base_seconds=1.0, max_seconds=60.0, random_value=0.0)
    jittered = compute_backoff(3, base_seconds=1.0, max_seconds=60.0, random_value=1.0)

    assert small == 1.0
    assert second == 2.0
    assert tenth == 60.0, "must cap"
    assert jittered > compute_backoff(3, base_seconds=1.0, max_seconds=60.0, random_value=0.0)


async def test_worker_triages_an_ingested_event(store, settings):
    from triage import RuleBasedTriageEngine

    await store.save_sla(make_sla())
    await store.ingest_event(make_event(), raw_payload="{}", max_attempts=3)
    sent: list = []
    worker = build_worker(store, settings, RuleBasedTriageEngine(), sent=sent)
    await worker.start()
    try:
        decision = await wait_for_decision(store, "EVT-1")
    finally:
        await worker.stop()

    assert decision is not None
    assert decision.action.value in {"AUTO_EMAIL", "HUMAN_ESCALATION", "NO_ACTION"}
    assert (await store.outbox_stats())["DONE"] == 1
    assert worker.stats.completed == 1


async def test_a_crashed_worker_does_not_lose_the_job(store, settings):
    """Simulate a worker dying mid-triage: the lease expires and the job comes back."""
    from triage import RuleBasedTriageEngine

    await store.save_sla(make_sla())
    await store.ingest_event(make_event(), raw_payload="{}", max_attempts=3)

    abandoned = await store.claim_jobs(limit=1, lease_seconds=1)
    assert len(abandoned) == 1, "the doomed worker took the job"
    assert await store.get_decision_for_event("EVT-1") is None

    await asyncio.sleep(1.05)  # the worker never came back

    worker = build_worker(store, settings, RuleBasedTriageEngine())
    await worker.start()
    try:
        decision = await wait_for_decision(store, "EVT-1")
    finally:
        await worker.stop()

    assert decision is not None, "the reclaimed job must still produce a decision"
    assert worker.stats.recovered_leases == 1
    assert (await store.outbox_stats())["DEAD"] == 0


async def test_a_flaky_engine_is_retried_until_it_succeeds(store, settings):
    from schemas import Severity, TriageAction

    class FlakyEngine(StubEngine):
        def __init__(self):
            super().__init__(None)
            self.failures = 2

        async def propose(self, context):
            self.calls += 1
            if self.failures > 0:
                self.failures -= 1
                raise TriageEngineError("provider hiccup", retryable=True)
            from schemas import TriageProposal

            return TriageProposal(
                severity=Severity.MEDIUM,
                action=TriageAction.NO_ACTION,
                confidence_score=0.9,
                reasoning_summary="recovered",
            )

    await store.save_sla(make_sla())
    settings.job_max_attempts = 5
    await store.ingest_event(make_event(), raw_payload="{}", max_attempts=5)
    engine = FlakyEngine()
    worker = build_worker(store, settings, engine)

    await worker.start()
    try:
        decision = await wait_for_decision(store, "EVT-1")
    finally:
        await worker.stop()

    assert decision is not None and decision.action.value == "NO_ACTION"
    assert engine.calls == 3, f"two failures then success, got {engine.calls}"
    assert worker.stats.retried == 2
    assert worker.stats.dead_lettered == 0


async def test_an_unexpected_exception_dead_letters_immediately(store, settings):
    class ExplodingEngine(StubEngine):
        async def propose(self, context):
            raise ValueError("a bug, not a provider problem")

    await store.save_sla(make_sla())
    await store.ingest_event(make_event(), raw_payload="{}", max_attempts=5)
    worker = build_worker(store, settings, ExplodingEngine())

    await worker.start()
    try:
        for _ in range(200):
            if (await store.outbox_stats())["DEAD"] == 1:
                break
            await asyncio.sleep(0.02)
    finally:
        await worker.stop()

    stats = await store.outbox_stats()
    assert stats["DEAD"] == 1, "a bug should surface immediately, not be retried into the ground"
    assert stats["DONE"] == 0
    assert worker.stats.dead_lettered == 1
    assert [row["kind"] for row in await store.list_dead_letters()] == ["JOB_EXHAUSTED"]


async def test_a_broken_engine_escalates_and_completes_the_job(store, settings):
    """A permanent engine failure is a decision (escalate), not a dead letter."""
    await store.save_sla(make_sla())
    await store.ingest_event(make_event(), raw_payload="{}", max_attempts=3)
    engine = StubEngine(error=TriageEngineError("bad output", retryable=False))
    worker = build_worker(store, settings, engine)

    await worker.start()
    try:
        decision = await wait_for_decision(store, "EVT-1")
    finally:
        await worker.stop()

    assert decision is not None and decision.action.value == "HUMAN_ESCALATION"
    assert (await store.outbox_stats())["DONE"] == 1
    assert await store.get_escalation(decision.decision_id) is not None


async def test_auto_email_is_actually_dispatched_and_recorded(store, settings):
    from triage import RuleBasedTriageEngine

    await store.save_sla(make_sla())
    await store.ingest_event(make_event(reported_delay_minutes=90), raw_payload="{}", max_attempts=3)
    sent: list = []
    worker = build_worker(store, settings, RuleBasedTriageEngine(), sent=sent)

    await worker.start()
    try:
        decision = await wait_for_decision(store, "EVT-1")
        for _ in range(200):
            rows = await store.get_notifications(decision.decision_id) if decision else []
            if rows and rows[0]["status"] == NotificationStatus.SENT.value:
                break
            await asyncio.sleep(0.02)
    finally:
        await worker.stop()

    assert decision is not None and decision.action.value == "AUTO_EMAIL"
    rows = await store.get_notifications(decision.decision_id)
    assert rows[0]["status"] == NotificationStatus.SENT.value
    assert sent and sent[0]["recipient"] == "ops@acme.example"


async def test_shutdown_never_leaves_a_job_stuck_in_flight(store, settings):
    from triage import RuleBasedTriageEngine

    await store.save_sla(make_sla())
    for index in range(5):
        await store.ingest_event(
            make_event(event_id=f"EVT-{index}", idempotency_key=f"key-{index}"),
            raw_payload="{}",
            max_attempts=3,
        )
    worker = build_worker(store, settings, RuleBasedTriageEngine())

    await worker.start()
    await asyncio.sleep(0.15)  # let it pick work up
    await worker.stop()

    stats = await store.outbox_stats()
    assert stats["CLAIMED"] == 0, "no lease may be left dangling after a clean shutdown"
    assert stats["DONE"] + stats["PENDING"] + stats["DEAD"] == 5
