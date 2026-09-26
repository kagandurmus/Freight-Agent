"""Persistence tests: the guarantees that must hold under concurrency and crashes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from schemas import ShipmentSLA
from store import IngestOutcome, JobStatus, to_db_time, utcnow
from conftest import make_event, make_sla


async def test_ingest_is_idempotent_under_concurrency(store):
    event = make_event()
    results = await asyncio.gather(
        *(store.ingest_event(event, raw_payload="{}", max_attempts=3) for _ in range(20))
    )
    accepted = [r for r in results if r.outcome is IngestOutcome.ACCEPTED]
    duplicates = [r for r in results if r.outcome is IngestOutcome.DUPLICATE]

    assert len(accepted) == 1, "the unique index must let exactly one writer win"
    assert len(duplicates) == 19
    assert await store.pending_job_count() == 1, "a redelivery must not enqueue a second job"


async def test_claiming_a_job_is_exclusive(store):
    for index in range(6):
        # Distinct idempotency keys: without them these six payloads fingerprint identically
        # and collapse into one event, which is the documented dedupe behaviour.
        result = await store.ingest_event(
            make_event(event_id=f"EVT-{index}", idempotency_key=f"key-{index}"),
            raw_payload="{}",
            max_attempts=3,
        )
        assert result.outcome is IngestOutcome.ACCEPTED

    claimed: list[str] = []

    async def claimer() -> None:
        for _ in range(6):
            jobs = await store.claim_jobs(limit=1, lease_seconds=60)
            claimed.extend(job.job_id for job in jobs)
            await asyncio.sleep(0)

    await asyncio.gather(*(claimer() for _ in range(4)))

    assert len(claimed) == 6, f"each job claimed exactly once, got {len(claimed)}"
    assert len(set(claimed)) == 6, "no job may be claimed by two workers"


async def test_expired_lease_is_reclaimed(store):
    await store.ingest_event(make_event(), raw_payload="{}", max_attempts=3)
    first = await store.claim_jobs(limit=5, lease_seconds=1)
    assert len(first) == 1 and first[0].attempts == 1
    assert await store.claim_jobs(limit=5, lease_seconds=1) == [], "lease still held"

    later = utcnow() + timedelta(seconds=5)
    reclaimed = await store.claim_jobs(limit=5, lease_seconds=60, now=later)
    assert len(reclaimed) == 1
    assert reclaimed[0].job_id == first[0].job_id
    assert reclaimed[0].attempts == 2, "a reclaimed job counts as a retry"


async def test_release_expired_leases_returns_jobs_to_the_pool(store):
    await store.ingest_event(make_event(), raw_payload="{}", max_attempts=3)
    await store.claim_jobs(limit=5, lease_seconds=1)

    released = await store.release_expired_leases(now=utcnow() + timedelta(seconds=5))
    assert released == 1
    assert (await store.outbox_stats())["PENDING"] == 1


async def test_failed_jobs_back_off_then_dead_letter(store):
    await store.ingest_event(make_event(), raw_payload="{}", max_attempts=2)

    job = (await store.claim_jobs(limit=1, lease_seconds=60))[0]
    assert await store.fail_job(job, error="transient", retry_in_seconds=30) is JobStatus.PENDING
    row = await store.db.fetch_one("SELECT next_attempt_at, attempts FROM outbox WHERE job_id = ?", (job.job_id,))
    assert row["attempts"] == 1
    assert row["next_attempt_at"] > to_db_time(utcnow()), "the retry must be deferred"

    # Not due yet.
    assert await store.claim_jobs(limit=1, lease_seconds=60) == []

    later = utcnow() + timedelta(seconds=60)
    job2 = (await store.claim_jobs(limit=1, lease_seconds=60, now=later))[0]
    assert await store.fail_job(job2, error="permanent", retry_in_seconds=30) is JobStatus.DEAD
    assert (await store.outbox_stats())["DEAD"] == 1
    dead = await store.list_dead_letters()
    assert dead[0]["kind"] == "JOB_EXHAUSTED"


async def test_reused_event_id_is_a_conflict_not_an_overwrite(store):
    await store.ingest_event(make_event(), raw_payload="{}", max_attempts=3)
    revised = make_event(reported_delay_minutes=999)  # same event_id, new content

    result = await store.ingest_event(revised, raw_payload="{}", max_attempts=3)

    assert result.outcome is IngestOutcome.CONFLICT
    stored = await store.get_event("EVT-1")
    assert stored is not None and stored.reported_delay_minutes == 90, "history must not be rewritten"
    assert [row["kind"] for row in await store.list_dead_letters()] == ["EVENT_ID_REUSED"]


async def test_stored_timestamps_sort_chronologically(store):
    early = datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=UTC)
    late = datetime(2026, 1, 1, 0, 0, 0, 500_000, tzinfo=UTC)
    assert to_db_time(early) < to_db_time(late)
    assert to_db_time(early) <= to_db_time(early)
    with pytest.raises(ValueError):
        to_db_time(datetime(2026, 1, 1))  # naive


async def test_sla_resolution_picks_the_version_in_force(store):
    base = make_sla(customer_tier="STANDARD", max_allowable_delay_minutes=60)
    later_version = make_sla(
        customer_tier="VIP",
        max_allowable_delay_minutes=30,
        effective_from=datetime(2026, 6, 1, tzinfo=UTC),
    ).model_copy(update={"version": 2})
    await store.save_sla(base)
    await store.save_sla(later_version)

    before = await store.resolve_sla("SHP-1", datetime(2026, 1, 1, tzinfo=UTC))
    after = await store.resolve_sla("SHP-1", datetime(2026, 7, 1, tzinfo=UTC))

    assert before is not None and before.customer_tier.value == "STANDARD"
    assert after is not None and after.version == 2 and after.customer_tier.value == "VIP"
    assert await store.resolve_sla("SHP-UNKNOWN", datetime(2026, 7, 1, tzinfo=UTC)) is None


async def test_saving_the_same_decision_twice_is_a_no_op(store):
    """A lease expiry can let a second worker finish the same job; only one may act."""
    from schemas import Severity, TriageAction, TriageResult

    sla = make_sla()
    event = make_event()
    await store.ingest_event(event, raw_payload="{}", max_attempts=3)
    job = (await store.claim_jobs(limit=1, lease_seconds=60))[0]
    assessment = sla.assess(event, now=event.occurred_at)
    from schemas import EmailDraft

    result = TriageResult(
        event_id="EVT-1",
        shipment_id="SHP-1",
        assessment=assessment,
        severity=Severity.MEDIUM,
        action=TriageAction.AUTO_EMAIL,
        confidence_score=0.9,
        email_draft=EmailDraft(to=["ops@acme.example"], subject="Delay", body="Body"),
        reasoning_summary="first writer",
    )

    assert await store.save_decision(result, job_id=job.job_id, latency_ms=5) is True
    assert await store.save_decision(result, job_id=job.job_id, latency_ms=5) is False
    assert len(await store.get_notifications(result.decision_id)) == 1
    assert (await store.outbox_stats())["DONE"] == 1


async def test_awaiting_approval_is_invisible_to_automatic_dispatch(store):
    """The whole point of the status split: automation cannot see a human-gated draft."""
    from schemas import EmailDraft, Severity, TriageAction, TriageResult

    sla = make_sla()
    event = make_event()
    await store.ingest_event(event, raw_payload="{}", max_attempts=3)
    assessment = sla.assess(event, now=event.occurred_at)
    result = TriageResult(
        event_id="EVT-1",
        shipment_id="SHP-1",
        assessment=assessment,
        severity=Severity.HIGH,
        action=TriageAction.HUMAN_ESCALATION,
        confidence_score=0.9,
        email_draft=EmailDraft(to=["ops@acme.example"], subject="Delay", body="Body"),
        escalation_reason="needs a human",
        reasoning_summary="gated",
    )
    await store.save_decision(result, job_id=None)

    assert await store.pending_notifications() == [], "dispatchable queue must stay empty"

    released = await store.approve_notifications(result.decision_id)
    assert released == 1
    assert len(await store.pending_notifications()) == 1


async def test_notifications_are_deduplicated_per_recipient(store):
    from schemas import EmailDraft, Severity, TriageAction, TriageResult

    sla = make_sla()
    event = make_event()
    await store.ingest_event(event, raw_payload="{}", max_attempts=3)
    assessment = sla.assess(event, now=event.occurred_at)
    result = TriageResult(
        event_id="EVT-1",
        shipment_id="SHP-1",
        assessment=assessment,
        severity=Severity.MEDIUM,
        action=TriageAction.AUTO_EMAIL,
        confidence_score=0.9,
        email_draft=EmailDraft(
            to=["ops@acme.example", "ops@acme.example"],
            cc=["risk@acme.example"],
            subject="Delay notice",
            body="B"
        ),
        reasoning_summary="dedupe",
    )
    await store.save_decision(result, job_id=None)

    rows = await store.get_notifications(result.decision_id)
    assert sorted(row["recipient"] for row in rows) == ["ops@acme.example", "risk@acme.example"]
