"""HTTP surface for the Freight Exception Triage service.

Three contracts shape this module:

* **Ingestion never does triage.** `POST /webhooks/carrier-delay` validates, writes the event
  and its outbox job in one transaction, wakes a worker and answers 202. A carrier's webhook
  timeout is not spent waiting on a language model.
* **Every rejection is classified and preserved.** Malformed JSON, a schema violation and a
  reused primary key are three different failures; all three are dead-lettered with the raw
  body so the event can be replayed after the producer fixes it. Nothing is silently dropped.
* **Backpressure is explicit.** Above `max_pending_jobs` the endpoint returns 503 with
  Retry-After, because accepting work we cannot drain is how queues become outages.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import ValidationError

from config import Settings
from llm import LLMTriageEngine
from observability import configure_logging
from schemas import DelayEventWebhook, WebhookAck
from seed import seed_demo_slas
from store import Database, IngestOutcome, Store, utcnow
from triage import (
    BaseTriageEngine,
    NotificationDispatcher,
    RuleBasedTriageEngine,
    TriageService,
)
from worker import TriageWorker

__all__ = ["build_engine", "create_app", "running_client"]

logger = logging.getLogger(__name__)


def build_engine(settings: Settings) -> BaseTriageEngine:
    """Select the triage engine.

    Misconfiguration fails at startup rather than at the first carrier alert: a service that
    boots happily and then escalates every delay because nobody set an API key is worse than
    one that refuses to start.
    """
    if settings.triage_engine != "llm":
        return RuleBasedTriageEngine()

    providers = settings.to_provider_chain()
    if not providers:
        raise RuntimeError(
            "TRIAGE_ENGINE=llm requires at least one provider credential: set "
            "DEEPSEEK_API_KEY (primary) and optionally OPENAI_API_KEY (fallback), or use "
            "TRIAGE_ENGINE=rule_based."
        )
    return LLMTriageEngine(
        providers,
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        max_validation_retries=settings.llm_max_validation_retries,
        prompt_version=settings.prompt_version,
    )


def _validation_summary(exc: ValidationError) -> list[dict[str, Any]]:
    """JSON-safe error list; Pydantic's raw errors can carry unserialisable context."""
    return [
        {
            "field": ".".join(str(part) for part in error["loc"]) or "<body>",
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors(include_url=False, include_context=False, include_input=False)
    ]


def create_app(
    settings: Settings | None = None, *, engine: BaseTriageEngine | None = None
) -> FastAPI:
    """Composition root: build the graph, own its lifecycle.

    @@engine@@ is an injection seam: tests and the live-LLM runner supply a wrapper around the
    real engine to observe what the model actually proposed, while the app still owns
    start/close. When it is None the engine is selected from settings as usual.
    """
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, json_logs=settings.log_json)

        db = Database(
            settings.database_path,
            pool_size=settings.db_pool_size,
            busy_timeout_ms=settings.db_busy_timeout_ms,
        )
        await db.start()
        schema_version = await db.migrate()
        store = Store(db)

        if settings.seed_demo_slas and await store.count_slas() == 0:
            seeded = await seed_demo_slas(store)
            logger.info("seeded %s demo service level agreements", seeded)

        active_engine = engine if engine is not None else build_engine(settings)
        await active_engine.start()
        dispatcher = NotificationDispatcher(store)
        service = TriageService(store, active_engine, settings.to_policy())
        worker = TriageWorker(
            store=store, service=service, dispatcher=dispatcher, settings=settings
        )

        app.state.settings = settings
        app.state.db = db
        app.state.store = store
        app.state.engine = active_engine
        app.state.dispatcher = dispatcher
        app.state.worker = worker
        app.state.schema_version = schema_version

        worst_case_triage = (
            settings.llm_timeout_seconds
            * settings.llm_max_attempts
            * max(1, len(settings.to_provider_chain()))
        )
        if settings.job_lease_seconds <= worst_case_triage:
            logger.warning(
                "job lease (%ss) is not longer than the worst-case triage time (%ss): a slow "
                "model call can outlive its lease and be processed twice",
                settings.job_lease_seconds,
                worst_case_triage,
            )

        logger.info(
            "started %s env=%s engine=%s db=%s schema=v%s",
            settings.app_name,
            settings.environment,
            active_engine.name,
            settings.database_path,
            schema_version,
        )
        await worker.start()
        try:
            yield
        finally:
            await worker.stop()
            await active_engine.close()
            await db.close()
            logger.info("shutdown complete")

    app = FastAPI(
        title="Freight Exception Triage",
        version="0.1.0",
        summary="Carrier delay alerts in, auditable SLA triage decisions out.",
        lifespan=lifespan,
    )

    # -- ingestion --------------------------------------------------------------------

    @app.post("/webhooks/carrier-delay", status_code=202)
    async def ingest_carrier_delay(request: Request, response: Response) -> dict[str, Any]:
        """Accept a carrier delay alert for asynchronous triage."""
        store: Store = request.app.state.store
        raw_body = (await request.body()).decode("utf-8", errors="replace")

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            await store.record_dead_letter(
                kind="MALFORMED_JSON", reference=None, reason=str(exc), payload=raw_body
            )
            raise HTTPException(status_code=400, detail=f"body is not valid JSON: {exc}") from exc

        try:
            event = DelayEventWebhook.model_validate(payload)
        except ValidationError as exc:
            errors = _validation_summary(exc)
            await store.record_dead_letter(
                kind="INVALID_EVENT",
                reference=str(payload.get("event_id") or payload.get("idempotency_key") or "unknown"),
                reason=json.dumps(errors),
                payload=raw_body,
            )
            raise HTTPException(status_code=422, detail=errors) from exc

        if await store.pending_job_count() >= request.app.state.settings.max_pending_jobs:
            # Refusing work we cannot drain beats growing an unbounded backlog.
            raise HTTPException(
                status_code=503,
                detail="triage backlog is full; retry shortly",
                headers={"Retry-After": "5"},
            )

        result = await store.ingest_event(
            event,
            raw_payload=raw_body,
            max_attempts=request.app.state.settings.job_max_attempts,
        )
        if result.outcome is IngestOutcome.CONFLICT:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"event_id {result.event_id} already exists with different content; "
                    "use a new event_id for a revised report"
                ),
            )

        if result.outcome is IngestOutcome.ACCEPTED:
            request.app.state.worker.notify()  # durable already: the row is committed
        else:
            response.status_code = 200
            logger.info("duplicate delivery of %s ignored", result.idempotency_key)

        ack = WebhookAck(
            accepted=True,
            event_id=result.event_id,
            idempotency_key=result.idempotency_key,
            duplicate=result.outcome is IngestOutcome.DUPLICATE,
            message=(
                "queued for asynchronous triage"
                if result.outcome is IngestOutcome.ACCEPTED
                else "already received; no new work queued"
            ),
        )
        return ack.model_dump(mode="json")

    # -- decisions --------------------------------------------------------------------

    @app.get("/decisions/{decision_id}")
    async def get_decision(decision_id: str, request: Request) -> dict[str, Any]:
        decision = await request.app.state.store.get_decision(decision_id)
        if decision is None:
            raise HTTPException(status_code=404, detail="unknown decision_id")
        return decision.model_dump(mode="json")

    @app.get("/events/{event_id}")
    async def get_event(event_id: str, request: Request) -> dict[str, Any]:
        """The canonicalised event plus the untouched carrier body, for audits and replay."""
        store: Store = request.app.state.store
        event = await store.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="unknown event_id")
        return {
            "event": event.model_dump(mode="json"),
            "raw_payload": await store.get_event_raw(event_id),
        }

    @app.get("/events/{event_id}/decision")
    async def get_decision_for_event(event_id: str, request: Request) -> dict[str, Any]:
        """Operator question number one: 'what happened to this alert?'"""
        decision = await request.app.state.store.get_decision_for_event(event_id)
        if decision is None:
            raise HTTPException(status_code=404, detail="no decision yet for this event")
        return decision.model_dump(mode="json")

    @app.get("/decisions")
    async def list_decisions(
        request: Request, shipment_id: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        rows = await request.app.state.store.list_decisions(
            shipment_id=shipment_id, limit=max(1, min(limit, 200))
        )
        return {"count": len(rows), "decisions": rows}

    @app.get("/escalations")
    async def list_escalations(request: Request, limit: int = 50) -> dict[str, Any]:
        rows = await request.app.state.store.open_escalations(limit=max(1, min(limit, 200)))
        return {"count": len(rows), "escalations": rows}

    @app.get("/notifications/{decision_id}")
    async def list_notifications(decision_id: str, request: Request) -> dict[str, Any]:
        rows = await request.app.state.store.get_notifications(decision_id)
        return {"count": len(rows), "notifications": rows}

    @app.post("/operator/decisions/{decision_id}/approve")
    async def approve_decision(decision_id: str, request: Request) -> dict[str, Any]:
        """Operator sign-off: release human-approved drafts and close the escalation.

        This is the only path that can move an `AWAITING_APPROVAL` draft. The automated
        dispatcher cannot see those rows at all.
        """
        store: Store = request.app.state.store
        decision = await store.get_decision(decision_id)
        if decision is None:
            raise HTTPException(status_code=404, detail="unknown decision_id")
        released = await store.approve_notifications(decision_id)
        dispatched = await request.app.state.dispatcher.dispatch_for_decision(decision_id)
        return {
            "decision_id": decision_id,
            "released": released,
            "dispatched": dispatched,
            "action": decision.action.value,
        }

    @app.get("/dead-letters")
    async def list_dead_letters(request: Request, limit: int = 50) -> dict[str, Any]:
        store: Store = request.app.state.store
        rows = await store.list_dead_letters(limit=max(1, min(limit, 200)))
        return {"count": len(rows), "total": await store.count_dead_letters(), "dead_letters": rows}

    # -- operations -------------------------------------------------------------------

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        store: Store = request.app.state.store
        return {
            "status": "ok",
            "schema_version": request.app.state.schema_version,
            "engine": request.app.state.engine.name,
            "db_ok": await store.db.scalar("SELECT 1") == 1,
            "outbox": await store.outbox_stats(),
            "backlog": await store.pending_job_count(),
            "dead_letters": await store.count_dead_letters(),
            "workers": request.app.state.worker.stats.as_dict(),
            "server_time": utcnow().isoformat(),
        }

    @app.get("/")
    async def index() -> dict[str, Any]:
        return {
            "service": "freight-exception-triage",
            "endpoints": {
                "ingest": "POST /webhooks/carrier-delay",
                "event": "GET /events/{event_id}",
                "decision": "GET /decisions/{decision_id}",
                "decision_for_event": "GET /events/{event_id}/decision",
                "decisions": "GET /decisions?shipment_id=",
                "escalations": "GET /escalations",
                "notifications": "GET /notifications/{decision_id}",
                "approve": "POST /operator/decisions/{decision_id}/approve",
                "dead_letters": "GET /dead-letters",
                "health": "GET /healthz",
                "docs": "/docs",
            },
        }

    return app


@asynccontextmanager
async def running_client(
    app: FastAPI, *, base_url: str = "http://triage.local", **client_kwargs: Any
) -> AsyncIterator[httpx.AsyncClient]:
    """Run an app's lifespan and hand back an HTTP client bound to it over ASGI.

    `httpx.ASGITransport` deliberately does not run startup/shutdown, so without this the
    demo and the tests would post webhooks into an app with no database and no workers -- and
    pass, for the wrong reason. Driving `lifespan_context` explicitly is the same thing
    Starlette's own test client does, minus the thread.
    """
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url=base_url, **client_kwargs
        ) as client:
            yield client
