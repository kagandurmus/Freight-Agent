"""Shared fixtures.

Each test gets its own SQLite file and its own application graph: no global state, no
ordering dependencies, and the worker pool is real unless a test deliberately supplies a stub
engine.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from config import Settings
from schemas import DelayEventWebhook, ShipmentSLA, TriageProposal
from store import Database, Store
from triage import BaseTriageEngine, TriageContext

NOW = datetime.now(UTC)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "triage.db"


@pytest.fixture
async def database(db_path: Path) -> AsyncIterator[Database]:
    db = Database(db_path, pool_size=4)
    await db.start()
    await db.migrate()
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture
async def store(database: Database) -> Store:
    return Store(database)


@pytest.fixture
def settings(db_path: Path) -> Settings:
    return Settings(
        database_path=db_path,
        worker_count=2,
        worker_poll_interval_seconds=0.1,
        job_lease_seconds=30,
        job_backoff_base_seconds=0.05,
        job_backoff_max_seconds=0.2,
        shutdown_grace_seconds=5.0,
        log_level="WARNING",
        seed_demo_slas=False,
    )


def make_event(**overrides: Any) -> DelayEventWebhook:
    """A valid, unremarkable delay report; override whatever the test cares about."""
    payload: dict[str, Any] = {
        "event_id": "EVT-1",
        "carrier_id": "DBSC",
        "shipment_id": "SHP-1",
        "occurred_at": NOW - timedelta(minutes=5),
        "reported_delay_minutes": 90,
        "reason": "WEATHER",
    }
    payload.update(overrides)
    return DelayEventWebhook.model_validate(payload)


def make_sla(**overrides: Any) -> ShipmentSLA:
    terms: dict[str, Any] = {
        "shipment_id": "SHP-1",
        "customer_tier": "STANDARD",
        "max_allowable_delay_minutes": 60,
        "penalty_per_hour": "100.00",
        "notification_emails": ["ops@acme.example"],
    }
    terms.update(overrides)
    return ShipmentSLA.model_validate(terms)


class StubEngine(BaseTriageEngine):
    """Engine that returns a fixed proposal, or raises, on demand.

    Lives here rather than in triage.py because it exists to test the service's behaviour
    *around* an engine, not to be a usable engine.
    """

    name = "stub-engine"
    prompt_version = "stub-v1"

    def __init__(
        self,
        proposal: TriageProposal | None = None,
        *,
        error: Exception | None = None,
        delay_seconds: float = 0.0,
        gate: asyncio.Event | None = None,
    ):
        self.proposal = proposal
        self.error = error
        self.delay_seconds = delay_seconds
        self.gate = gate
        self.calls = 0

    async def propose(self, context: TriageContext) -> TriageProposal:
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        assert self.proposal is not None, "StubEngine needs a proposal or an error"
        return self.proposal


@pytest.fixture
def stub_proposal() -> TriageProposal:
    from schemas import EmailDraft, Severity, TriageAction

    return TriageProposal(
        severity=Severity.MEDIUM,
        action=TriageAction.AUTO_EMAIL,
        confidence_score=0.95,
        email_draft=EmailDraft(to=["ops@acme.example"], subject="Delay", body="Body"),
        reasoning_summary="stub",
    )
