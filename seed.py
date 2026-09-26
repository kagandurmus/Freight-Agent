"""Demo contract data.

Kept out of the persistence layer on purpose: seeding is a fixture, not a property of the
store. The same helper is used by the demo runner and by the API when
`TRIAGE_SEED_DEMO_SLAS` is on, so a freshly started server has something to triage.
"""

from __future__ import annotations

from schemas import ShipmentSLA
from store import Store

__all__ = ["DEMO_SHIPMENTS", "seed_demo_slas"]

#: (shipment_id, tier, allowance, grace, penalty/hour, cap, recipients)
DEMO_SHIPMENTS: tuple[tuple[str, str, int, int, str, str | None, list[str]], ...] = (
    (
        "SHP-4471",  # VIP: a routine breach still trips the tier guardrail
        "VIP",
        60,
        15,
        "150.00",
        "900.00",
        ["ops@acme-logistics.example"],
    ),
    (
        "SHP-1001",  # STANDARD: the auto-email path
        "STANDARD",
        60,
        0,
        "50.00",
        "500.00",
        ["planning@nordwind.example"],
    ),
    (
        "SHP-1002",  # generous terms, so small delays are genuinely non-events
        "STANDARD",
        120,
        30,
        "25.00",
        None,
        ["planning@nordwind.example"],
    ),
)


async def seed_demo_slas(store: Store) -> int:
    """Insert the demo contracts; returns how many rows were written."""
    written = 0
    for shipment_id, tier, allowance, grace, rate, cap, recipients in DEMO_SHIPMENTS:
        sla = ShipmentSLA.from_tier(
            shipment_id,
            tier,
            penalty_per_hour=rate,
            notification_emails=recipients,
            max_allowable_delay_minutes=allowance,
            grace_period_minutes=grace,
            penalty_cap=cap,
            customer_id="ACME GmbH" if tier == "VIP" else "Nordwind AG",
        )
        await store.save_sla(sla)
        written += 1
    return written
