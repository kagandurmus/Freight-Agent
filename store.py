"""Asynchronous SQLite persistence for the Freight Exception Triage service.

Three ideas carry the weight here:

**1. The transactional outbox.** Accepting a webhook and triaging it must not be one atomic
operation -- triage calls a model and can take seconds. So ingestion writes the event *and*
its outbox job in a single SQLite transaction, commits, and returns 202. The job survives a
crash, a redeploy or a model outage because it is in the same durable write as the event.

**2. Lease-based claiming.** Workers claim jobs with an `UPDATE ... RETURNING` that flips
`status` to `CLAIMED` and sets `lease_until` in one statement, so two workers can never
claim the same job. A worker that dies mid-triage simply lets its lease expire and the job
is reclaimed -- no orphaned rows, no cleanup daemon.

**3. Idempotency in the schema, not in Python.** `UNIQUE` constraints on
`idempotency_key`, `decisions.event_id` and `(decision_id, channel, recipient)` mean a
retry cannot double-charge a customer or double-email a broker. Checking for existence in
application code would race; a unique index cannot.

Timestamps are stored as canonical UTC ISO-8601 with fixed microsecond precision
(`2026-01-01T00:00:00.000000+00:00`) so that lexicographic ordering in SQL equals
chronological ordering. Mixing offsets or dropping the microseconds would silently corrupt
every window query.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import asyncio

import aiosqlite

from schemas import DelayEventWebhook, ShipmentSLA, TriageAction, TriageResult, to_stored_json

__all__ = [
    "Database",
    "IngestOutcome",
    "IngestResult",
    "Job",
    "JobStatus",
    "NotificationStatus",
    "Store",
    "from_db_time",
    "to_db_time",
    "utcnow",
]


# --------------------------------------------------------------------------------------
# Time and json helpers
# --------------------------------------------------------------------------------------


def utcnow() -> datetime:
    """Single source of 'now' so tests can freeze time by monkeypatching one symbol."""
    return datetime.now(UTC)


def to_db_time(value: datetime) -> str:
    """Canonical, sortable, fixed-width UTC ISO-8601."""
    if value.tzinfo is None:
        raise ValueError("refusing to persist a naive datetime: it has no unambiguous instant")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def from_db_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------

MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        1,
        (
            """
            CREATE TABLE IF NOT EXISTS shipment_sla (
                sla_id         TEXT PRIMARY KEY,
                shipment_id    TEXT NOT NULL,
                version        INTEGER NOT NULL,
                customer_tier  TEXT NOT NULL,
                effective_from TEXT,
                effective_to   TEXT,
                payload        TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                UNIQUE (shipment_id, version)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_sla_lookup ON shipment_sla (shipment_id, effective_from, effective_to)",
            """
            CREATE TABLE IF NOT EXISTS delay_events (
                event_id                TEXT PRIMARY KEY,
                idempotency_key         TEXT NOT NULL UNIQUE,
                carrier_id              TEXT NOT NULL,
                shipment_id             TEXT NOT NULL,
                source                  TEXT NOT NULL,
                reason                  TEXT NOT NULL,
                occurred_at             TEXT NOT NULL,
                received_at             TEXT NOT NULL,
                reported_delay_minutes  INTEGER NOT NULL,
                effective_delay_minutes INTEGER NOT NULL,
                payload                 TEXT NOT NULL,
                raw_payload             TEXT NOT NULL,
                created_at              TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_events_shipment ON delay_events (shipment_id, created_at)",
            """
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id    TEXT PRIMARY KEY,
                event_id       TEXT NOT NULL UNIQUE REFERENCES delay_events(event_id),
                shipment_id    TEXT NOT NULL,
                severity       TEXT NOT NULL,
                action         TEXT NOT NULL,
                confidence     REAL NOT NULL,
                policy_version TEXT NOT NULL,
                model_name     TEXT,
                prompt_version TEXT,
                latency_ms     INTEGER,
                payload        TEXT NOT NULL,
                created_at     TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_decisions_shipment ON decisions (shipment_id, created_at)",
            """
            CREATE TABLE IF NOT EXISTS notifications (
                notification_id TEXT PRIMARY KEY,
                decision_id     TEXT NOT NULL REFERENCES decisions(decision_id),
                channel         TEXT NOT NULL,
                recipient       TEXT NOT NULL,
                subject         TEXT NOT NULL,
                body            TEXT NOT NULL,
                status          TEXT NOT NULL,
                attempts        INTEGER NOT NULL DEFAULT 0,
                last_error      TEXT,
                created_at      TEXT NOT NULL,
                sent_at         TEXT,
                UNIQUE (decision_id, channel, recipient)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications (status, created_at)",
            """
            CREATE TABLE IF NOT EXISTS escalations (
                escalation_id TEXT PRIMARY KEY,
                decision_id   TEXT NOT NULL UNIQUE REFERENCES decisions(decision_id),
                severity      TEXT NOT NULL,
                reason        TEXT NOT NULL,
                status        TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                resolved_at   TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS outbox (
                job_id          TEXT PRIMARY KEY,
                event_id        TEXT NOT NULL UNIQUE REFERENCES delay_events(event_id),
                kind            TEXT NOT NULL,
                status          TEXT NOT NULL,
                attempts        INTEGER NOT NULL DEFAULT 0,
                max_attempts    INTEGER NOT NULL,
                next_attempt_at TEXT NOT NULL,
                lease_until     TEXT,
                last_error      TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_outbox_claim ON outbox (status, next_attempt_at)",
            """
            CREATE TABLE IF NOT EXISTS dead_letters (
                dead_letter_id TEXT PRIMARY KEY,
                kind           TEXT NOT NULL,
                reference      TEXT,
                reason         TEXT NOT NULL,
                payload        TEXT NOT NULL,
                created_at     TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_dead_letters_kind ON dead_letters (kind, created_at)",
        ),
    ),
)


class JobStatus(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    DONE = "DONE"
    DEAD = "DEAD"


class IngestOutcome(StrEnum):
    """Three genuinely different answers to 'did you accept this webhook?'."""

    ACCEPTED = "ACCEPTED"
    #: Same idempotency key seen before: a redelivery. Safe to ignore, cheap to answer.
    DUPLICATE = "DUPLICATE"
    #: Same event_id with a *different* payload: the producer reused a primary key. We refuse
    #: rather than overwrite an audited event, and dead-letter it for a human to reconcile.
    CONFLICT = "CONFLICT"


class NotificationStatus(StrEnum):
    #: Queued for automatic dispatch (only ever set for a policy-approved AUTO_EMAIL).
    PENDING = "PENDING"
    #: Drafted for a human. Automation will never move this row on its own.
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    SENT = "SENT"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class IngestResult:
    outcome: IngestOutcome
    event_id: str
    idempotency_key: str
    job_id: str | None = None


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    event_id: str
    attempts: int
    max_attempts: int
    kind: str = "TRIAGE"


# --------------------------------------------------------------------------------------
# Connection pool
# --------------------------------------------------------------------------------------


class Database:
    """A tiny aiosqlite pool: one connection (and thread) per slot, WAL for readers.

    SQLite allows a single writer at a time, so write transactions are additionally
    serialised in-process by `_write_lock`. That converts lock contention into an ordered
    queue instead of a `SQLITE_BUSY` retry storm, while `busy_timeout` remains as the
    safety net for other processes touching the same file.
    """

    def __init__(self, path: Path | str, *, pool_size: int = 4, busy_timeout_ms: int = 5_000):
        self.path = str(path)
        self._pool_size = pool_size
        self._busy_timeout_ms = busy_timeout_ms
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._connections: list[aiosqlite.Connection] = []

    async def start(self) -> None:
        if self.path == ":memory:" and self._pool_size > 1:
            # Every aiosqlite connection would get its own private database, so migrations
            # run against one and queries read an empty one. Loud beats subtle.
            raise ValueError("an in-memory database requires pool_size=1")
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        for _ in range(self._pool_size):
            conn = await aiosqlite.connect(self.path, isolation_level=None)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
            self._connections.append(conn)
            self._pool.put_nowait(conn)

    async def close(self) -> None:
        for conn in self._connections:
            await conn.close()
        self._connections.clear()

    async def migrate(self) -> int:
        """Apply pending migrations; returns the resulting schema version."""
        async with self.transaction() as conn:
            current = (await (await conn.execute("PRAGMA user_version")).fetchone())[0]
            for version, statements in MIGRATIONS:
                if version <= current:
                    continue
                for statement in statements:
                    await conn.execute(statement)
                # PRAGMA cannot be parameterised; the value comes from our own tuple.
                await conn.execute(f"PRAGMA user_version={int(version)}")
                current = version
        return current

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[aiosqlite.Connection]:
        """Borrow a pooled connection for reads or single-statement writes."""
        conn = await self._pool.get()
        try:
            yield conn
        finally:
            self._pool.put_nowait(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Borrow a connection inside `BEGIN IMMEDIATE`, committing or rolling back.

        `IMMEDIATE` takes the write lock up front, so two concurrent ingests of the same
        idempotency key serialise here and the loser sees the unique-constraint conflict
        rather than a stale read.
        """
        async with self._write_lock:
            conn = await self._pool.get()
            try:
                await conn.execute("BEGIN IMMEDIATE")
                try:
                    yield conn
                except BaseException:
                    await conn.execute("ROLLBACK")
                    raise
                await conn.execute("COMMIT")
            finally:
                self._pool.put_nowait(conn)

    # -- read helpers -----------------------------------------------------------------

    async def fetch_one(
        self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> aiosqlite.Row | None:
        async with self.lease() as conn:
            cursor = await conn.execute(sql, params)
            try:
                return await cursor.fetchone()
            finally:
                await cursor.close()

    async def fetch_all(
        self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> list[aiosqlite.Row]:
        async with self.lease() as conn:
            cursor = await conn.execute(sql, params)
            try:
                return list(await cursor.fetchall())
            finally:
                await cursor.close()

    async def scalar(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> Any:
        row = await self.fetch_one(sql, params)
        return None if row is None else row[0]

    async def execute_write(
        self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> int:
        """Single-statement write; returns the number of affected rows."""
        async with self.transaction() as conn:
            cursor = await conn.execute(sql, params)
            try:
                affected = cursor.rowcount
            finally:
                await cursor.close()
        return max(0, affected)


# --------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------


class Store:
    """Domain queries. Every method is async and never blocks the event loop."""

    def __init__(self, db: Database):
        self.db = db

    # -- SLA resolution (Step 1, minimal) ---------------------------------------------

    async def save_sla(self, sla: ShipmentSLA) -> None:
        now = to_db_time(utcnow())
        async with self.db.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO shipment_sla
                    (sla_id, shipment_id, version, customer_tier,
                     effective_from, effective_to, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (shipment_id, version) DO UPDATE SET
                    customer_tier  = excluded.customer_tier,
                    effective_from = excluded.effective_from,
                    effective_to   = excluded.effective_to,
                    payload        = excluded.payload
                """,
                (
                    sla.sla_id,
                    sla.shipment_id,
                    sla.version,
                    sla.customer_tier.value,
                    to_db_time(sla.effective_from) if sla.effective_from else None,
                    to_db_time(sla.effective_to) if sla.effective_to else None,
                    to_stored_json(sla),
                    now,
                ),
            )

    async def resolve_sla(self, shipment_id: str, at: datetime) -> ShipmentSLA | None:
        """The highest-version terms in force at `at`, or None when nothing covers it."""
        moment = to_db_time(at)
        row = await self.db.fetch_one(
            """
            SELECT payload FROM shipment_sla
             WHERE shipment_id = ?
               AND (effective_from IS NULL OR effective_from <= ?)
               AND (effective_to   IS NULL OR effective_to   >  ?)
             ORDER BY version DESC
             LIMIT 1
            """,
            (shipment_id, moment, moment),
        )
        return None if row is None else ShipmentSLA.model_validate_json(row["payload"])

    async def count_slas(self) -> int:
        return int(await self.db.scalar("SELECT COUNT(*) FROM shipment_sla") or 0)

    # -- ingestion --------------------------------------------------------------------

    async def ingest_event(
        self,
        event: DelayEventWebhook,
        *,
        raw_payload: str,
        max_attempts: int,
        now: datetime | None = None,
    ) -> IngestResult:
        """Persist the event and enqueue its triage job in one transaction.

        Returns `ACCEPTED` for new work, `DUPLICATE` for a redelivery of something we
        already hold, or `CONFLICT` when the producer reused an `event_id` for different
        content -- the last case is dead-lettered rather than silently overwriting history.
        """
        moment = to_db_time(now or utcnow())
        async with self.db.transaction() as conn:
            cursor = await conn.execute(
                """
                INSERT OR IGNORE INTO delay_events
                    (event_id, idempotency_key, carrier_id, shipment_id, source, reason,
                     occurred_at, received_at, reported_delay_minutes,
                     effective_delay_minutes, payload, raw_payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.idempotency_key,
                    event.carrier_id,
                    event.shipment_id,
                    event.source.value,
                    event.reason.value,
                    to_db_time(event.occurred_at),
                    to_db_time(event.received_at),
                    event.reported_delay_minutes,
                    event.effective_delay_minutes,
                    to_stored_json(event),
                    raw_payload,
                    moment,
                ),
            )
            if cursor.rowcount == 0:
                existing = await self._existing_event(conn, event)
                return existing

            job_id = f"job_{uuid.uuid4().hex}"
            await conn.execute(
                """
                INSERT INTO outbox
                    (job_id, event_id, kind, status, attempts, max_attempts,
                     next_attempt_at, created_at, updated_at)
                VALUES (?, ?, 'TRIAGE', ?, 0, ?, ?, ?, ?)
                """,
                (job_id, event.event_id, JobStatus.PENDING.value, max_attempts, moment, moment, moment),
            )
            return IngestResult(
                outcome=IngestOutcome.ACCEPTED,
                event_id=event.event_id,
                idempotency_key=event.idempotency_key or "",
                job_id=job_id,
            )

    @staticmethod
    async def _existing_event(
        conn: aiosqlite.Connection, event: DelayEventWebhook
    ) -> IngestResult:
        """Classify a rejected insert as a benign redelivery or a genuine key collision."""
        cursor = await conn.execute(
            "SELECT event_id, idempotency_key FROM delay_events WHERE event_id = ?",
            (event.event_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is not None:
            if row["idempotency_key"] == event.idempotency_key:
                return IngestResult(
                    IngestOutcome.DUPLICATE, row["event_id"], row["idempotency_key"]
                )
            await conn.execute(
                """
                INSERT INTO dead_letters (dead_letter_id, kind, reference, reason, payload, created_at)
                VALUES (?, 'EVENT_ID_REUSED', ?, ?, ?, ?)
                """,
                (
                    f"dl_{uuid.uuid4().hex}",
                    event.event_id,
                    "event_id already exists with a different idempotency_key",
                    _json(
                        {
                            "incoming": json.loads(event.model_dump_json()),
                            "stored_idempotency_key": row["idempotency_key"],
                        }
                    ),
                    to_db_time(utcnow()),
                ),
            )
            return IngestResult(
                IngestOutcome.CONFLICT, event.event_id, event.idempotency_key or ""
            )

        cursor = await conn.execute(
            "SELECT event_id, idempotency_key FROM delay_events WHERE idempotency_key = ?",
            (event.idempotency_key,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is not None:  # same key, different event_id: still a redelivery
            return IngestResult(IngestOutcome.DUPLICATE, row["event_id"], row["idempotency_key"])
        raise RuntimeError(
            f"insert for {event.event_id} was ignored but no conflicting row exists"
        )

    # -- outbox -----------------------------------------------------------------------

    async def claim_jobs(
        self,
        *,
        limit: int,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> list[Job]:
        """Atomically claim up to `limit` due jobs, reclaiming expired leases.

        One statement, so it is safe across workers and processes: the `UPDATE ... RETURNING`
        is what makes the claim exclusive.
        """
        moment = now or utcnow()
        lease_until = moment + timedelta(seconds=lease_seconds)
        rows = await self.db.fetch_all(
            """
            UPDATE outbox
               SET status = :claimed,
                   lease_until = :lease_until,
                   attempts = attempts + 1,
                   updated_at = :now
             WHERE job_id IN (
                   SELECT job_id FROM outbox
                    WHERE (status = :pending AND next_attempt_at <= :now)
                       OR (status = :claimed AND lease_until IS NOT NULL AND lease_until < :now)
                    ORDER BY next_attempt_at
                    LIMIT :limit
             )
            RETURNING job_id, event_id, attempts, max_attempts, kind
            """,
            {
                "claimed": JobStatus.CLAIMED.value,
                "pending": JobStatus.PENDING.value,
                "lease_until": to_db_time(lease_until),
                "now": to_db_time(moment),
                "limit": limit,
            },
        )
        return [
            Job(
                job_id=row["job_id"],
                event_id=row["event_id"],
                attempts=row["attempts"],
                max_attempts=row["max_attempts"],
                kind=row["kind"],
            )
            for row in rows
        ]

    async def complete_job(self, job_id: str, *, now: datetime | None = None) -> None:
        async with self.db.transaction() as conn:
            await conn.execute(
                "UPDATE outbox SET status = ?, lease_until = NULL, updated_at = ? WHERE job_id = ?",
                (JobStatus.DONE.value, to_db_time(now or utcnow()), job_id),
            )

    async def fail_job(
        self,
        job: Job,
        *,
        error: str,
        retry_in_seconds: float,
        now: datetime | None = None,
    ) -> JobStatus:
        """Reschedule with backoff, or dead-letter once attempts are exhausted.

        Note `attempts` was already incremented at claim time, so a job that has just failed
        its final attempt really is out of attempts.
        """
        moment = now or utcnow()
        exhausted = job.attempts >= job.max_attempts
        async with self.db.transaction() as conn:
            if exhausted:
                await conn.execute(
                    """
                    UPDATE outbox
                       SET status = ?, lease_until = NULL, last_error = ?, updated_at = ?
                     WHERE job_id = ?
                    """,
                    (JobStatus.DEAD.value, error[:2_000], to_db_time(moment), job.job_id),
                )
                await conn.execute(
                    """
                    INSERT INTO dead_letters (dead_letter_id, kind, reference, reason, payload, created_at)
                    VALUES (?, 'JOB_EXHAUSTED', ?, ?, ?, ?)
                    """,
                    (
                        f"dl_{uuid.uuid4().hex}",
                        job.job_id,
                        error[:2_000],
                        _json({"job_id": job.job_id, "event_id": job.event_id, "attempts": job.attempts}),
                        to_db_time(moment),
                    ),
                )
                return JobStatus.DEAD

            retry_at = moment + timedelta(seconds=max(0.0, retry_in_seconds))
            await conn.execute(
                """
                UPDATE outbox
                   SET status = ?, lease_until = NULL, last_error = ?,
                       next_attempt_at = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (
                    JobStatus.PENDING.value,
                    error[:2_000],
                    to_db_time(retry_at),
                    to_db_time(moment),
                    job.job_id,
                ),
            )
            return JobStatus.PENDING

    async def release_expired_leases(self, *, now: datetime | None = None) -> int:
        """Startup hygiene: hand abandoned jobs back to the pool immediately.

        `claim_jobs` already treats an expired lease as claimable, so this is not required
        for correctness -- it makes recovery observable (and testable) instead of implicit.
        """
        moment = to_db_time(now or utcnow())
        cursor = await self.db.execute_write(
            """
            UPDATE outbox SET status = ?, lease_until = NULL, updated_at = ?
             WHERE status = ? AND lease_until IS NOT NULL AND lease_until < ?
            """,
            (JobStatus.PENDING.value, moment, JobStatus.CLAIMED.value, moment),
        )
        return cursor

    async def outbox_stats(self) -> dict[str, int]:
        rows = await self.db.fetch_all("SELECT status, COUNT(*) AS n FROM outbox GROUP BY status")
        stats = {status.value: 0 for status in JobStatus}
        for row in rows:
            stats[row["status"]] = row["n"]
        return stats

    async def pending_job_count(self) -> int:
        return int(
            await self.db.scalar(
                "SELECT COUNT(*) FROM outbox WHERE status IN (?, ?)",
                (JobStatus.PENDING.value, JobStatus.CLAIMED.value),
            )
            or 0
        )

    # -- events and decisions ---------------------------------------------------------

    async def get_event(self, event_id: str) -> DelayEventWebhook | None:
        row = await self.db.fetch_one("SELECT payload FROM delay_events WHERE event_id = ?", (event_id,))
        return None if row is None else DelayEventWebhook.model_validate_json(row["payload"])

    async def get_event_raw(self, event_id: str) -> str | None:
        row = await self.db.fetch_one("SELECT raw_payload FROM delay_events WHERE event_id = ?", (event_id,))
        return None if row is None else row["raw_payload"]

    async def save_decision(
        self,
        result: TriageResult,
        *,
        job_id: str | None,
        latency_ms: int | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Persist the decision, its side-effect rows and the job completion atomically.

        Returns False when a decision for this event already exists (a lease expiry allowed a
        second worker to finish the same job). The `INSERT OR IGNORE` on
        `decisions.event_id` is the guard: exactly one worker gets to write the customer-facing
        outcomes, and the other becomes a no-op rather than a duplicate email.
        """
        moment = to_db_time(now or utcnow())
        async with self.db.transaction() as conn:
            cursor = await conn.execute(
                """
                INSERT OR IGNORE INTO decisions
                    (decision_id, event_id, shipment_id, severity, action, confidence,
                     policy_version, model_name, prompt_version, latency_ms, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.decision_id,
                    result.event_id,
                    result.shipment_id,
                    result.severity.value,
                    result.action.value,
                    result.confidence_score,
                    result.policy_version,
                    result.model_name,
                    result.prompt_version,
                    latency_ms,
                    to_stored_json(result),
                    moment,
                ),
            )
            first_writer = cursor.rowcount == 1

            if first_writer and result.email_draft is not None:
                # Only a policy-approved AUTO_EMAIL is queued for automatic dispatch; a draft
                # attached to an escalation waits for a human, and stays waiting.
                status = (
                    NotificationStatus.PENDING
                    if result.action is TriageAction.AUTO_EMAIL
                    else NotificationStatus.AWAITING_APPROVAL
                )
                draft = result.email_draft
                for recipient in dict.fromkeys([*draft.to, *draft.cc]):
                    await conn.execute(
                        """
                        INSERT OR IGNORE INTO notifications
                            (notification_id, decision_id, channel, recipient, subject, body,
                             status, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"ntf_{uuid.uuid4().hex}",
                            result.decision_id,
                            draft.channel.value,
                            recipient,
                            draft.subject,
                            draft.body,
                            status.value,
                            moment,
                        ),
                    )
            if first_writer and result.action is TriageAction.HUMAN_ESCALATION:
                await conn.execute(
                    """
                    INSERT OR IGNORE INTO escalations
                        (escalation_id, decision_id, severity, reason, status, created_at)
                    VALUES (?, ?, ?, ?, 'OPEN', ?)
                    """,
                    (
                        f"esc_{uuid.uuid4().hex}",
                        result.decision_id,
                        result.severity.value,
                        result.escalation_reason or "escalated for human review",
                        moment,
                    ),
                )
            if job_id is not None:
                await conn.execute(
                    """
                    UPDATE outbox SET status = ?, lease_until = NULL, updated_at = ?
                     WHERE job_id = ?
                    """,
                    (JobStatus.DONE.value, moment, job_id),
                )
            return first_writer

    async def get_decision(self, decision_id: str) -> TriageResult | None:
        row = await self.db.fetch_one("SELECT payload FROM decisions WHERE decision_id = ?", (decision_id,))
        return None if row is None else TriageResult.model_validate_json(row["payload"])

    async def get_decision_for_event(self, event_id: str) -> TriageResult | None:
        row = await self.db.fetch_one("SELECT payload FROM decisions WHERE event_id = ?", (event_id,))
        return None if row is None else TriageResult.model_validate_json(row["payload"])

    async def list_decisions(self, *, shipment_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Compact decision summaries, newest first."""
        if shipment_id is None:
            rows = await self.db.fetch_all(
                """
                SELECT decision_id, event_id, shipment_id, severity, action, confidence,
                       policy_version, model_name, created_at
                  FROM decisions ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            )
        else:
            rows = await self.db.fetch_all(
                """
                SELECT decision_id, event_id, shipment_id, severity, action, confidence,
                       policy_version, model_name, created_at
                  FROM decisions WHERE shipment_id = ? ORDER BY created_at DESC LIMIT ?
                """,
                (shipment_id, limit),
            )
        return [dict(row) for row in rows]

    # -- notifications ----------------------------------------------------------------

    async def pending_notifications(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Rows eligible for automatic dispatch. AWAITING_APPROVAL is deliberately excluded."""
        rows = await self.db.fetch_all(
            """
            SELECT n.notification_id, n.decision_id, n.channel, n.recipient, n.subject, n.body
              FROM notifications n
             WHERE n.status = ?
             ORDER BY n.created_at
             LIMIT ?
            """,
            (NotificationStatus.PENDING.value, limit),
        )
        return [dict(row) for row in rows]

    async def mark_notification_sent(self, notification_id: str, *, error: str | None = None) -> None:
        moment = to_db_time(utcnow())
        async with self.db.transaction() as conn:
            if error is None:
                await conn.execute(
                    """
                    UPDATE notifications
                       SET status = ?, sent_at = ?, attempts = attempts + 1, last_error = NULL
                     WHERE notification_id = ?
                    """,
                    (NotificationStatus.SENT.value, moment, notification_id),
                )
            else:
                await conn.execute(
                    """
                    UPDATE notifications
                       SET attempts = attempts + 1, last_error = ?
                     WHERE notification_id = ?
                    """,
                    (error[:1_000], notification_id),
                )

    async def get_notifications(self, decision_id: str) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """
            SELECT notification_id, channel, recipient, subject, status, attempts, sent_at
              FROM notifications WHERE decision_id = ? ORDER BY created_at
            """,
            (decision_id,),
        )
        return [dict(row) for row in rows]

    async def approve_notifications(self, decision_id: str) -> int:
        """Operator action: release human-approved drafts for dispatch."""
        moment = to_db_time(utcnow())
        cursor = await self.db.execute_write(
            """
            UPDATE notifications SET status = ?
             WHERE decision_id = ? AND status = ?
            """,
            (
                NotificationStatus.PENDING.value,
                decision_id,
                NotificationStatus.AWAITING_APPROVAL.value,
            ),
        )
        async with self.db.transaction() as conn:
            await conn.execute(
                "UPDATE escalations SET status = 'RESOLVED', resolved_at = ? WHERE decision_id = ?",
                (moment, decision_id),
            )
        return cursor

    async def get_escalation(self, decision_id: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            "SELECT escalation_id, decision_id, severity, reason, status, created_at FROM escalations WHERE decision_id = ?",
            (decision_id,),
        )
        return None if row is None else dict(row)

    async def open_escalations(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """
            SELECT e.escalation_id, e.decision_id, d.shipment_id, e.severity, e.reason, e.created_at
              FROM escalations e JOIN decisions d ON d.decision_id = e.decision_id
             WHERE e.status = 'OPEN' ORDER BY e.created_at DESC LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]

    # -- dead letters -----------------------------------------------------------------

    async def record_dead_letter(
        self, *, kind: str, reference: str | None, reason: str, payload: str
    ) -> str:
        dead_letter_id = f"dl_{uuid.uuid4().hex}"
        async with self.db.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO dead_letters (dead_letter_id, kind, reference, reason, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (dead_letter_id, kind, reference, reason[:2_000], payload[:20_000], to_db_time(utcnow())),
            )
        return dead_letter_id

    async def list_dead_letters(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            "SELECT dead_letter_id, kind, reference, reason, created_at FROM dead_letters ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in rows]

    async def count_dead_letters(self) -> int:
        return int(await self.db.scalar("SELECT COUNT(*) FROM dead_letters") or 0)
