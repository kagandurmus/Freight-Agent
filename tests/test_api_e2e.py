"""End-to-end HTTP tests: the contracts the carrier and the operator actually see."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

import api
from api import create_app, running_client
from conftest import StubEngine, make_event, make_sla
from schemas import to_stored_json


def wire(event) -> dict[str, Any]:
    """The JSON a carrier would actually send (declared fields only, no computed values)."""
    return json.loads(to_stored_json(event))


async def wait_decision(
    client, event_id: str, *, timeout: float = 5.0
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get(f"/events/{event_id}/decision")
        if response.status_code == 200:
            return response.json()
        await asyncio.sleep(0.02)
    return None


@pytest.fixture
def app_settings(settings):
    settings.seed_demo_slas = False
    settings.worker_count = 2
    settings.worker_poll_interval_seconds = 0.05
    settings.shutdown_grace_seconds = 2.0
    return settings


async def test_routine_breach_notifies_the_customer_over_http(app_settings):
    app = create_app(app_settings)
    async with running_client(app) as client:
        await app.state.store.save_sla(make_sla())
        response = await client.post(
            "/webhooks/carrier-delay", json=wire(make_event(reported_delay_minutes=90))
        )
        assert response.status_code == 202
        ack = response.json()
        assert ack["accepted"] is True and ack["duplicate"] is False
        assert ack["idempotency_key"].startswith("auto_")

        decision = await wait_decision(client, "EVT-1")
        assert decision is not None
        assert decision["action"] == "AUTO_EMAIL"
        notifications = (await client.get(f"/notifications/{decision['decision_id']}")).json()
        assert notifications["notifications"][0]["status"] == "SENT"


async def test_a_redelivered_alert_is_deduped(app_settings):
    app = create_app(app_settings)
    async with running_client(app) as client:
        await app.state.store.save_sla(make_sla())
        payload = wire(make_event())
        first = await client.post("/webhooks/carrier-delay", json=payload)
        second = await client.post("/webhooks/carrier-delay", json=payload)

        assert first.status_code == 202
        assert second.status_code == 200
        assert second.json()["duplicate"] is True
        assert second.json()["event_id"] == first.json()["event_id"]
        await wait_decision(client, "EVT-1")
        listing = (await client.get("/decisions")).json()
        assert listing["count"] == 1, "a redelivery must not produce a second decision"


async def test_ingestion_is_never_blocked_by_a_slow_model(app_settings, monkeypatch, stub_proposal):
    """The whole reason triage is asynchronous."""
    gate = asyncio.Event()
    engine = StubEngine(stub_proposal, gate=gate)
    monkeypatch.setattr(api, "build_engine", lambda _settings: engine)
    app_settings.max_pending_jobs = 1

    app = create_app(app_settings)
    async with running_client(app) as client:
        await app.state.store.save_sla(make_sla())
        first = await client.post("/webhooks/carrier-delay", json=wire(make_event(event_id="EVT-1")))
        assert first.status_code == 202, "accepted even though the model is hanging"

        second = await client.post("/webhooks/carrier-delay", json=wire(make_event(event_id="EVT-2")))
        assert second.status_code == 503
        assert second.headers["retry-after"] == "5", "backpressure must tell the carrier when to retry"

        gate.set()  # release the model call so the app can shut down cleanly
        assert await wait_decision(client, "EVT-1") is not None


async def test_a_human_gated_draft_needs_an_operator_to_move(app_settings):
    app = create_app(app_settings)
    async with running_client(app) as client:
        await app.state.store.save_sla(make_sla())
        # 120 min against a 60 min allowance -> ratio 1.0 -> HIGH -> escalate with a draft.
        await client.post(
            "/webhooks/carrier-delay", json=wire(make_event(reported_delay_minutes=120))
        )
        decision = await wait_decision(client, "EVT-1")
        assert decision is not None and decision["action"] == "HUMAN_ESCALATION"

        notifications = (await client.get(f"/notifications/{decision['decision_id']}")).json()
        assert notifications["notifications"][0]["status"] == "AWAITING_APPROVAL"

        escalations = (await client.get("/escalations")).json()
        assert escalations["count"] == 1

        approval = await client.post(f"/operator/decisions/{decision['decision_id']}/approve")
        assert approval.status_code == 200
        assert approval.json()["released"] == 1

        after = (await client.get(f"/notifications/{decision['decision_id']}")).json()
        assert after["notifications"][0]["status"] == "SENT"
        assert (await client.get("/escalations")).json()["count"] == 0, "approval closes it"


async def test_unknown_ids_are_404_not_500(app_settings):
    app = create_app(app_settings)
    async with running_client(app) as client:
        assert (await client.get("/decisions/trg_nope")).status_code == 404
        assert (await client.get("/events/EVT-nope")).status_code == 404
        assert (await client.get("/events/EVT-nope/decision")).status_code == 404
        assert (
            await client.post("/operator/decisions/trg_nope/approve")
        ).status_code == 404


async def test_every_rejection_is_classified_and_preserved(app_settings):
    app = create_app(app_settings)
    async with running_client(app) as client:
        await app.state.store.save_sla(make_sla())

        invalid = await client.post("/webhooks/carrier-delay", json={"event_id": "EVT-BAD"})
        assert invalid.status_code == 422
        assert any(error["field"] == "carrier_id" for error in invalid.json()["detail"])

        malformed = await client.post(
            "/webhooks/carrier-delay", content=b"{oops", headers={"content-type": "application/json"}
        )
        assert malformed.status_code == 400

        await client.post("/webhooks/carrier-delay", json=wire(make_event()))
        conflict = await client.post(
            "/webhooks/carrier-delay", json=wire(make_event(reported_delay_minutes=999))
        )
        assert conflict.status_code == 409, "a reused event_id must not overwrite history"

        dead = (await client.get("/dead-letters")).json()
        kinds = {row["kind"] for row in dead["dead_letters"]}
        assert kinds == {"INVALID_EVENT", "MALFORMED_JSON", "EVENT_ID_REUSED"}


async def test_the_original_carrier_body_is_preserved_verbatim(app_settings):
    app = create_app(app_settings)
    async with running_client(app) as client:
        await app.state.store.save_sla(make_sla())
        payload = {**wire(make_event()), "trailer_plate": "M-AB-1234", "unmodelled": {"nested": True}}
        await client.post("/webhooks/carrier-delay", json=payload)

        stored = (await client.get("/events/EVT-1")).json()
        assert stored["event"]["unmapped_payload_keys"] == ["trailer_plate", "unmodelled"]
        assert json.loads(stored["raw_payload"])["unmodelled"] == {"nested": True}


async def test_healthz_exposes_pipeline_state(app_settings):
    app = create_app(app_settings)
    async with running_client(app) as client:
        await app.state.store.save_sla(make_sla())
        await client.post("/webhooks/carrier-delay", json=wire(make_event()))
        await wait_decision(client, "EVT-1")

        health = (await client.get("/healthz")).json()
        assert health["status"] == "ok"
        assert health["db_ok"] is True
        assert health["schema_version"] == 1
        assert health["outbox"]["DONE"] == 1
        assert health["workers"]["completed"] == 1


async def test_the_index_lists_the_surface(app_settings):
    app = create_app(app_settings)
    async with running_client(app) as client:
        body = (await client.get("/")).json()
        assert "POST /webhooks/carrier-delay" in body["endpoints"]["ingest"]
