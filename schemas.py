"""Typed contracts for the Freight Exception Triage service.

When a carrier reports a delay we must answer three questions, in this order:

1. **Does this breach the customer SLA?**  -> deterministic arithmetic, never an LLM.
2. **How bad is it, and who should act?**  -> graded severity + routing decision.
3. **What exactly do we tell the broker?**  -> a reviewable draft or an escalation.

Design principles
-----------------
1. *Deterministic facts in code, narrative from the model.*  Breach, excess minutes and
   penalty exposure are computed by `ShipmentSLA.assess`. An LLM may only grade severity,
   explain itself and draft prose. A hallucinated euro figure must be impossible by
   construction, so the money lives in `SLAAssessment`, not in free text.
2. *Fail safe, never fail silent.*  Contradictory decisions are rejected by validators and
   the service falls back to `TriageResult.safe_fallback` (human escalation), never to
   an un-reviewed customer e-mail.
3. *Lenient at the edge, strict in the core.*  `DelayEventWebhook` ingests messy carrier
   payloads (unknown reasons, extra keys, missing idempotency key) and flags them;
   `ShipmentSLA` / `TriageResult` are `extra="forbid"` contracts where a typo is a bug.
4. *Everything is auditable.*  Every result carries event/shipment/SLA linkage, the
   assessment it was derived from, model + prompt + policy versions and a timestamp, so a
   decision can be replayed and explained months later.
5. *Money is Decimal, time is timezone-aware, free text is untrusted.*  Floats lose cents;
   naive datetimes silently shift by hours; `driver_notes` is attacker-influenced input
   that must never be concatenated into a system prompt.

Review deltas vs. the Gemini draft (v1 prototype)
-------------------------------------------------
Fixed gaps
  * `action` had no `NO_ACTION`: a delay that does *not* breach the SLA still forced a
    customer e-mail or a human escalation. Breach is now computable and routable.
  * `TriageResult` had no invariants, so `severity=CRITICAL, action=AUTO_EMAIL,
    confidence=0.2` validated happily. Cross-field rules now reject that.
  * `email_draft: str | None` cannot express recipients or a subject; replaced by the
    structured `EmailDraft` (recipient list, subject, body, language, channel).
  * `ShipmentSLA` hardcoded EUR and had no grace period, penalty cap, penalty accrual
    basis, effective window, or notification recipients.
  * Nothing linked a result back to its inputs, so no audit trail existed (`event_id`,
    `sla_id`, `assessment`, `model_name`, `prompt_version`, `policy_version`).
Hardened
  * `reported_delay_minutes` is bounded, not any int; `confidence_score` rejects NaN and
    infinities, not merely out-of-range numbers; timestamps must be timezone-aware.
  * `idempotency_key` is now optional with a deterministic fingerprint fallback, and its
    semantics vs. `event_id` are documented (retry dedupe vs. identity).
  * Money is `Decimal` with 2dp and non-finite protection.
Added
  * ETA-derived delay cross-check (`derived_delay_minutes` / `effective_delay_minutes`):
    carriers under-report, so the worst of the two is used and the gap becomes a signal.
  * `EventSource` reliability priors, `TriagePolicy` guardrails, structured
    `TriageFlag`s, `WebhookAck`, and a conservative `safe_fallback`.
  * Prompt-injection heuristics on carrier free text: flagged, never silently trusted,
    never dropped.

Python 3.11+ (`StrEnum`), Pydantic v2.  Deliberately dependency-free: e-mail validation is
pragmatic and local; swap `EmailAddress` for `pydantic.EmailStr` once `pydantic[email]`
is a declared dependency.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Annotated, Any, Final, TypeVar

from pydantic import (
    AfterValidator,
    AliasChoices,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

__all__ = [
    "CarrierId",
    "ConfidenceScore",
    "CurrencyCode",
    "CustomerTier",
    "DelayEventWebhook",
    "DelayReason",
    "EmailAddress",
    "EmailDraft",
    "EventId",
    "EventSource",
    "Money",
    "NotificationChannel",
    "PenaltyBasis",
    "SCHEMA_VERSION",
    "SLAAssessment",
    "Severity",
    "ShipmentId",
    "ShipmentSLA",
    "TriageAction",
    "TriageFlag",
    "TriagePolicy",
    "TriageProposal",
    "TriageResult",
    "WebhookAck",
    "revalidated",
    "severity_from_assessment",
    "to_stored_json",
]

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

SCHEMA_VERSION: Final[str] = "1.0"

#: Anything beyond two weeks is not a delay, it is a data-quality incident.
MAX_PLAUSIBLE_DELAY_MINUTES: Final[int] = 20_160
#: Tolerance for carrier clocks running ahead of ours.
CLOCK_SKEW_TOLERANCE: Final[timedelta] = timedelta(minutes=10)
#: Replayed events older than this are dead-lettered rather than retried.
MAX_EVENT_AGE: Final[timedelta] = timedelta(days=30)
#: Carrier free text is truncated (never rejected) at the ingestion boundary.
FREE_TEXT_MAX_LENGTH: Final[int] = 2_000

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Bidi overrides / zero-width characters: invisible text that hides instructions from humans.
_INVISIBLE_CHARS = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
# Deliberately crude heuristic. A guardrail that raises a flag, not a security boundary.
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions?"
    r"|disregard\s+.{0,24}instructions?"
    r"|system\s+prompt"
    r"|you\s+are\s+now\s+"
    r"|new\s+instructions?\s*:"
    r"|<\s*/?\s*(system|assistant|tool)\s*>)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------------------
# Enumerations (wire values are UPPER_SNAKE, matching the v1 contract)
# --------------------------------------------------------------------------------------


class CustomerTier(StrEnum):
    """Commercial importance of the shipper; drives both allowance and escalation."""

    STANDARD = "STANDARD"
    VIP = "VIP"


class DelayReason(StrEnum):
    """Normalised delay cause. Drives severity weighting and how we message the broker."""

    WEATHER = "WEATHER"
    MECHANICAL_BREAKDOWN = "MECHANICAL_BREAKDOWN"
    TRAFFIC_CONGESTION = "TRAFFIC_CONGESTION"
    DRIVER_ISSUE = "DRIVER_ISSUE"
    ACCIDENT = "ACCIDENT"
    BORDER_OR_CUSTOMS = "BORDER_OR_CUSTOMS"
    CUSTOMER_CAUSED = "CUSTOMER_CAUSED"
    CAPACITY_OR_REASSIGNMENT = "CAPACITY_OR_REASSIGNMENT"
    ROAD_CLOSURE = "ROAD_CLOSURE"
    OTHER = "OTHER"


class EventSource(StrEnum):
    """Where the alert came from; sets the confidence prior for automated handling."""

    CARRIER_API = "CARRIER_API"
    EDI_214 = "EDI_214"
    DRIVER_APP = "DRIVER_APP"
    TMS_POLL = "TMS_POLL"
    CARRIER_PORTAL_EMAIL = "CARRIER_PORTAL_EMAIL"
    OPERATOR_MANUAL = "OPERATOR_MANUAL"


class Severity(StrEnum):
    """How loudly this exception should be treated internally."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TriageAction(StrEnum):
    """What the service is allowed to do without a human."""

    #: Log and stop. The correct answer for most non-breaching alerts.
    NO_ACTION = "NO_ACTION"
    #: Send the pre-generated customer notification automatically.
    AUTO_EMAIL = "AUTO_EMAIL"
    #: Put a drafted decision in front of an operator; nothing leaves the building.
    HUMAN_ESCALATION = "HUMAN_ESCALATION"


class NotificationChannel(StrEnum):
    EMAIL = "EMAIL"
    SMS = "SMS"
    EDI_214 = "EDI_214"
    WEBHOOK = "WEBHOOK"
    IN_APP = "IN_APP"


class PenaltyBasis(StrEnum):
    """Contractual accrual granularity -- a frequent source of billing disputes."""

    #: Penalty accrues pro-rata per minute (smoother, cheaper for the carrier).
    PER_MINUTE = "PER_MINUTE"
    #: Every started hour is billed in full (the common contract).
    PER_STARTED_HOUR = "PER_STARTED_HOUR"


class TriageFlag(StrEnum):
    """Structured, machine-countable reasons attached to a decision."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    #: The model asked for a lower severity than the arithmetic supports; the floor won.
    SEVERITY_BELOW_FLOOR = "SEVERITY_BELOW_FLOOR"
    #: No contractual terms on file, so conservative defaults were applied instead.
    SLA_NOT_FOUND = "SLA_NOT_FOUND"
    SEVERITY_TOO_HIGH_FOR_AUTOMATION = "SEVERITY_TOO_HIGH_FOR_AUTOMATION"
    VIP_REQUIRES_HUMAN = "VIP_REQUIRES_HUMAN"
    PENALTY_ABOVE_AUTO_LIMIT = "PENALTY_ABOVE_AUTO_LIMIT"
    DELAY_DISCREPANCY = "DELAY_DISCREPANCY"
    MISSING_ETA_DATA = "MISSING_ETA_DATA"
    SLA_NOT_BREACHED = "SLA_NOT_BREACHED"
    UNKNOWN_DELAY_REASON = "UNKNOWN_DELAY_REASON"
    UNTRUSTED_FREE_TEXT = "UNTRUSTED_FREE_TEXT"
    SUSPECTED_PROMPT_INJECTION = "SUSPECTED_PROMPT_INJECTION"
    VALIDATION_FALLBACK = "VALIDATION_FALLBACK"
    POLICY_DOWNGRADE = "POLICY_DOWNGRADE"


#: Default contractual terms per tier, used by `ShipmentSLA.from_tier`.
TIER_DEFAULT_ALLOWANCE_MINUTES: Final[dict[CustomerTier, int]] = {
    CustomerTier.STANDARD: 60,
    CustomerTier.VIP: 30,
}
TIER_DEFAULT_PENALTY_PER_HOUR: Final[dict[CustomerTier, Decimal]] = {
    CustomerTier.STANDARD: Decimal("50.00"),
    CustomerTier.VIP: Decimal("150.00"),
}
#: Internal prior on how far an automated decision should trust a given channel.
SOURCE_RELIABILITY_PRIOR: Final[dict[EventSource, float]] = {
    EventSource.CARRIER_API: 0.90,
    EventSource.EDI_214: 0.85,
    EventSource.DRIVER_APP: 0.80,
    EventSource.TMS_POLL: 0.75,
    EventSource.OPERATOR_MANUAL: 0.70,
    EventSource.CARRIER_PORTAL_EMAIL: 0.60,
}

_REASON_ALIASES: Final[dict[str, DelayReason]] = {
    "SNOW": DelayReason.WEATHER,
    "STORM": DelayReason.WEATHER,
    "FOG": DelayReason.WEATHER,
    "ICE": DelayReason.WEATHER,
    "BAD_WEATHER": DelayReason.WEATHER,
    "BREAKDOWN": DelayReason.MECHANICAL_BREAKDOWN,
    "MECHANICAL": DelayReason.MECHANICAL_BREAKDOWN,
    "TRUCK_BREAKDOWN": DelayReason.MECHANICAL_BREAKDOWN,
    "TYRE": DelayReason.MECHANICAL_BREAKDOWN,
    "TRAFFIC": DelayReason.TRAFFIC_CONGESTION,
    "CONGESTION": DelayReason.TRAFFIC_CONGESTION,
    "JAM": DelayReason.TRAFFIC_CONGESTION,
    "HOS": DelayReason.DRIVER_ISSUE,
    "DRIVER": DelayReason.DRIVER_ISSUE,
    "SICKNESS": DelayReason.DRIVER_ISSUE,
    "NO_SHOW": DelayReason.DRIVER_ISSUE,
    "COLLISION": DelayReason.ACCIDENT,
    "CUSTOMS": DelayReason.BORDER_OR_CUSTOMS,
    "BORDER": DelayReason.BORDER_OR_CUSTOMS,
    "CLEARANCE": DelayReason.BORDER_OR_CUSTOMS,
    "DOCK": DelayReason.CUSTOMER_CAUSED,
    "WAITING_AT_DOCK": DelayReason.CUSTOMER_CAUSED,
    "REFUSED_SLOT": DelayReason.CUSTOMER_CAUSED,
    "CUSTOMER": DelayReason.CUSTOMER_CAUSED,
    "CAPACITY": DelayReason.CAPACITY_OR_REASSIGNMENT,
    "REASSIGNMENT": DelayReason.CAPACITY_OR_REASSIGNMENT,
    "NO_TRUCK": DelayReason.CAPACITY_OR_REASSIGNMENT,
    "CLOSURE": DelayReason.ROAD_CLOSURE,
    "ROADWORKS": DelayReason.ROAD_CLOSURE,
}

#: Longest phrase first, so "TRUCK_BREAKDOWN" wins over "BREAKDOWN" on the same text.
_ALIASES_BY_SPECIFICITY: Final[tuple[str, ...]] = tuple(
    sorted(_REASON_ALIASES, key=lambda alias: (-len(alias.split("_")), alias))
)


def _resolve_reason(raw: str) -> DelayReason | None:
    """Best-effort mapping of a carrier's reason string onto our enum.

    Exact member and exact alias matches are tried first. Carrier portal and EDI payloads
    are usually prose rather than codes ("truck breakdown on the A3"), so the last resort is
    a phrase scan over the normalised tokens. Matching on token boundaries -- not raw
    substrings -- keeps short aliases from firing inside unrelated words ("ICE" in
    "SERVICE", "JAM" in "JAMAICA"). Returns None when nothing matches.
    """
    key = re.sub(r"[^A-Za-z0-9]+", "_", raw.strip().upper()).strip("_")
    if not key:
        return None
    if key in DelayReason.__members__:
        return DelayReason[key]
    if key in _REASON_ALIASES:
        return _REASON_ALIASES[key]
    tokens = key.split("_")
    for alias in _ALIASES_BY_SPECIFICITY:
        parts = alias.split("_")
        span = len(parts)
        if any(tokens[i : i + span] == parts for i in range(len(tokens) - span + 1)):
            return _REASON_ALIASES[alias]
    return None

_SEVERITY_BUMP: Final[dict[Severity, Severity]] = {
    Severity.LOW: Severity.LOW,
    Severity.MEDIUM: Severity.HIGH,
    Severity.HIGH: Severity.CRITICAL,
    Severity.CRITICAL: Severity.CRITICAL,
}

# --------------------------------------------------------------------------------------
# Reusable annotated types
# --------------------------------------------------------------------------------------

#: Anything the model reports about its own certainty. NaN/inf are rejected, not clamped.
ConfidenceScore = Annotated[
    float,
    Field(ge=0.0, le=1.0, allow_inf_nan=False, description="Model self-assessed certainty, 0.0-1.0"),
]

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$"

EventId = Annotated[str, Field(min_length=1, max_length=64, pattern=_ID_PATTERN)]
ShipmentId = Annotated[str, Field(min_length=1, max_length=64, pattern=_ID_PATTERN)]
CarrierId = Annotated[str, Field(min_length=2, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")]
CurrencyCode = Annotated[str, Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")]


def _validate_money(value: Any) -> Any:
    """Reject non-finite / sub-cent input before any numeric comparison runs."""
    if isinstance(value, Decimal):
        amount = value
    else:
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(f"invalid monetary amount: {value!r}") from exc
    if not amount.is_finite():
        raise ValueError("monetary amount must be finite (NaN/Infinity rejected)")
    exponent = amount.as_tuple().exponent
    if isinstance(exponent, int) and -exponent > 2:
        raise ValueError("monetary amount must have at most 2 decimal places")
    return amount


#: Money is Decimal so cents survive the round trip. Note: JSON output is a *string*
#: (e.g. "45.50") because that is the only lossless JSON representation of Decimal.
Money = Annotated[
    Decimal,
    BeforeValidator(_validate_money),
    Field(ge=0, max_digits=14, decimal_places=2),
]

#: Pragmatic, dependency-free address check: one @, a dotted domain, no header-injection
#: characters, lower-cased for dedupe. Replace with `pydantic.EmailStr` (needs
#: `pydantic[email]`) if deliverability-level validation is ever required.
_EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9!#$%&'*+/=?^_\x60{|}~.-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)


def _validate_email(value: str) -> str:
    address = value.strip()
    if any(char in address for char in '\r\n\t ,;<>"'):
        raise ValueError(f"e-mail address contains an illegal character: {value!r}")
    if len(address) > 254 or not _EMAIL_PATTERN.match(address):
        raise ValueError(f"not a usable e-mail address: {value!r}")
    return address.lower()


EmailAddress = Annotated[str, AfterValidator(_validate_email)]

T = TypeVar("T", bound=BaseModel)


def _to_payload(value: Any) -> Any:
    """Rebuild a validation payload from *declared fields only*.

    `model_dump()` would include `computed_field` output, which then fails to re-validate
    against an `extra="forbid"` model. Walking the declared fields keeps `revalidated()`
    usable on models that expose computed values.
    """
    if isinstance(value, BaseModel):
        return {name: _to_payload(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, (list, tuple)):
        return [_to_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_payload(item) for key, item in value.items()}
    return value


def revalidated(model: T, **updates: Any) -> T:
    """Return a copy of `model` with `updates` applied **through validation**.

    `model_copy(update=...)` skips validators, which would let a policy override produce a
    structurally invalid decision record. This helper cannot.
    """
    payload = _to_payload(model)
    payload.update(updates)
    return type(model).model_validate(payload)


def to_stored_json(model: BaseModel) -> str:
    """Serialise **declared fields only**, so the result can be validated back later.

    @@model_dump_json()@@ also emits @@computed_field@@ values. Re-validating that output
    either fails outright (on an @@extra="forbid"@@ model such as @@ShipmentSLA@@, where the
    round trip raises on @@allowance_minutes@@) or, worse, silently succeeds on a lenient
    model and pollutes @@model_extra@@ -- which is how our own computed field names ended up
    being reported as unmapped carrier keys. Anything persisted and re-validated goes
    through this function.
    """
    return model.model_dump_json(exclude=set(type(model).model_computed_fields))


# --------------------------------------------------------------------------------------
# Inbound contract
# --------------------------------------------------------------------------------------


class DelayEventWebhook(BaseModel):
    """A carrier delay alert as received on the wire.

    This is the untrusted boundary of the service, so it is deliberately permissive:
    unknown keys are retained, unknown reasons degrade to `OTHER` (with the original
    string preserved in `reason_detail`), and an absent idempotency key is derived
    rather than rejected. Nothing here should raise a 422 that loses a real event, except
    genuine data corruption (impossible timestamps, absurd delays).

    `event_id` vs `idempotency_key`: `event_id` names *this* event revision and is what
    we join on downstream. `idempotency_key` is the producer's promise that a redelivery of
    the *same* alert carries the *same* key, so it is what we dedupe on. A corrected alert
    (new delay figure, same event) is intentionally a new fingerprint.
    """

    model_config = ConfigDict(
        extra="allow",  # never drop carrier fields we do not model yet
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    schema_version: str = Field(default=SCHEMA_VERSION, max_length=16)
    event_id: EventId
    idempotency_key: str | None = Field(default=None, max_length=128)
    carrier_id: CarrierId = Field(
        validation_alias=AliasChoices("carrier_id", "carrier_scac", "scac", "carrier")
    )
    shipment_id: ShipmentId
    source: EventSource = EventSource.CARRIER_API
    occurred_at: AwareDatetime = Field(
        validation_alias=AliasChoices("occurred_at", "timestamp", "event_time", "reported_at"),
        serialization_alias="occurred_at",
        description="When the delay was observed by the carrier. Timezone-aware, mandatory.",
    )
    received_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Server-side ingestion time; latency and clock skew are measured against it.",
    )
    reported_delay_minutes: int = Field(
        ge=0,
        le=MAX_PLAUSIBLE_DELAY_MINUTES,
        description="Carrier's own figure. Negative values are not delays and are rejected; "
        "an early-arrival update is a separate event type.",
    )
    reason: DelayReason = DelayReason.OTHER
    reason_detail: str | None = None
    driver_notes: str | None = None
    original_eta: AwareDatetime | None = None
    revised_eta: AwareDatetime | None = None
    current_location: str | None = Field(default=None, max_length=120)
    driver_id: str | None = Field(default=None, max_length=64)

    # -- canonicalisation -------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _normalise_reason(cls, data: Any) -> Any:
        """Map carrier-specific reason codes onto our enum without ever rejecting one."""
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        raw = payload.get("reason")
        if isinstance(raw, str):
            resolved = _resolve_reason(raw)
            if resolved is None:
                payload["reason"] = DelayReason.OTHER
                payload.setdefault("reason_detail", raw.strip())
            else:
                payload["reason"] = resolved
        else:
            payload["reason"] = raw or DelayReason.OTHER
        return payload

    @field_validator("reason_detail", "driver_notes", mode="before")
    @classmethod
    def _sanitise_free_text(cls, value: Any) -> Any:
        """Strip control/invisible characters and truncate.

        Untrusted input: kept verbatim enough to be useful, but never allowed to smuggle
        hidden instructions into a prompt or invisible text past a human reviewer.
        """
        if not isinstance(value, str):
            return value
        cleaned = _INVISIBLE_CHARS.sub("", _CONTROL_CHARS.sub("", value)).strip()
        return cleaned[:FREE_TEXT_MAX_LENGTH] or None

    @model_validator(mode="after")
    def _finalise(self) -> DelayEventWebhook:
        if self.occurred_at > self.received_at + CLOCK_SKEW_TOLERANCE:
            raise ValueError(
                "occurred_at is in the future beyond the clock-skew tolerance "
                f"({CLOCK_SKEW_TOLERANCE}); check the carrier clock before retrying"
            )
        if self.received_at - self.occurred_at > MAX_EVENT_AGE:
            raise ValueError(f"occurred_at is older than {MAX_EVENT_AGE.days} days: stale replay")
        if (
            self.original_eta is not None
            and self.revised_eta is not None
            and self.revised_eta <= self.original_eta
        ):
            raise ValueError(
                "revised_eta must be later than original_eta: this payload describes no delay"
            )
        if not self.idempotency_key:
            self.idempotency_key = self.fingerprint()
        return self

    # -- derived facts ----------------------------------------------------------------

    def fingerprint(self) -> str:
        """Deterministic dedupe key, used when the producer supplies none.

        Deliberately covers the *structured* identity of a report -- who, which shipment, when,
        how long, through which channel and why. Free text is excluded because a redelivery
        often reflows a note, and including it would defeat the dedupe. @@source@@ and @@reason@@
        are included because two alerts that differ only in those fields are genuinely
        different reports (an API alert and a portal e-mail are not the same event), and this
        fallback is only ever consulted when the producer sent no idempotency key at all.
        """
        material = "|".join(
            (
                self.schema_version,
                self.carrier_id,
                self.shipment_id,
                self.occurred_at.astimezone(UTC).isoformat(),
                str(self.reported_delay_minutes),
                self.source.value,
                self.reason.value,
            )
        )
        return "auto_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def derived_delay_minutes(self) -> int | None:
        """Delay recomputed from the ETA pair; None when the carrier sent no ETAs.

        Rounded up: a partial minute is still a minute of delay.
        """
        if self.original_eta is None or self.revised_eta is None:
            return None
        return max(0, math.ceil((self.revised_eta - self.original_eta).total_seconds() / 60))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_delay_minutes(self) -> int:
        """The delay we plan against: the **larger** of reported and ETA-derived.

        Carriers routinely under-report. When both numbers exist and disagree, the ETA pair
        is arithmetic rather than testimony, so the worst case wins and the gap is surfaced
        as `delay_discrepancy_minutes`.
        """
        derived = self.derived_delay_minutes
        if derived is None:
            return self.reported_delay_minutes
        return max(self.reported_delay_minutes, derived)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def delay_discrepancy_minutes(self) -> int | None:
        """Derived minus reported; positive means the carrier under-reported."""
        derived = self.derived_delay_minutes
        return None if derived is None else derived - self.reported_delay_minutes

    @computed_field  # type: ignore[prop-decorator]
    @property
    def base_confidence_prior(self) -> float:
        """How far an automated decision should be allowed to trust this channel."""
        return SOURCE_RELIABILITY_PRIOR.get(self.source, 0.50)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def injection_suspected(self) -> bool:
        """Heuristic guardrail over carrier free text.

        Flagged, never enforced: flagged text is still forwarded to humans and still used
        as context, but it must be delimited as data in any model prompt, and a human must
        see it before anything is sent.
        """
        haystack = f"{self.driver_notes or ''}\n{self.reason_detail or ''}"
        return bool(_INJECTION_PATTERNS.search(haystack))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unmapped_payload_keys(self) -> list[str]:
        """Carrier keys we did not model -- a living spec for the next schema revision."""
        return sorted(self.model_extra or {})


# --------------------------------------------------------------------------------------
# Contract terms
# --------------------------------------------------------------------------------------


class ShipmentSLA(BaseModel):
    """The contractual terms that apply to one shipment.

    Instances are versioned and time-bounded: a triage run must use the terms that were in
    force when the delay occurred, so `assess` refuses to evaluate against an SLA whose
    effective window does not contain the assessment time.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sla_id: str = Field(default_factory=lambda: f"sla_{uuid.uuid4().hex[:12]}", max_length=64)
    version: int = Field(default=1, ge=1)
    shipment_id: ShipmentId
    customer_id: str | None = Field(default=None, max_length=64)
    customer_tier: CustomerTier
    max_allowable_delay_minutes: int = Field(ge=0, le=MAX_PLAUSIBLE_DELAY_MINUTES)
    grace_period_minutes: int = Field(
        default=0,
        ge=0,
        le=MAX_PLAUSIBLE_DELAY_MINUTES,
        description="Free minutes on top of the allowance.",
    )
    penalty_basis: PenaltyBasis = PenaltyBasis.PER_STARTED_HOUR
    penalty_per_hour: Money = Decimal("0.00")
    penalty_cap: Money | None = Field(
        default=None,
        description="Contractual ceiling; must share the currency of the hourly rate.",
    )
    currency: CurrencyCode = "EUR"
    notification_emails: list[EmailAddress] = Field(default_factory=list, max_length=20)
    auto_notify_enabled: bool = Field(
        default=True, description="Master switch: False forces every decision to a human."
    )
    effective_from: AwareDatetime | None = None
    effective_to: AwareDatetime | None = None

    @model_validator(mode="after")
    def _check_terms(self) -> ShipmentSLA:
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_from >= self.effective_to
        ):
            raise ValueError("effective_from must be earlier than effective_to")
        if self.auto_notify_enabled and not self.notification_emails:
            raise ValueError(
                "auto_notify_enabled requires at least one notification address, otherwise "
                "AUTO_EMAIL decisions have nowhere to go"
            )
        return self

    @classmethod
    def from_tier(
        cls,
        shipment_id: str,
        customer_tier: CustomerTier | str,
        *,
        penalty_per_hour: Decimal | str | None = None,
        notification_emails: list[str] | None = None,
        **overrides: Any,
    ) -> ShipmentSLA:
        """Build an SLA from tier defaults, for onboarding and tests.

        Explicit keyword arguments always beat the tier defaults: onboarding a customer
        whose negotiated terms differ from the standard tier must not require hand-building
        the whole object.
        """
        tier = CustomerTier(customer_tier)
        terms: dict[str, Any] = {
            "shipment_id": shipment_id,
            "customer_tier": tier,
            "max_allowable_delay_minutes": TIER_DEFAULT_ALLOWANCE_MINUTES[tier],
            "penalty_per_hour": (
                TIER_DEFAULT_PENALTY_PER_HOUR[tier]
                if penalty_per_hour is None
                else Decimal(str(penalty_per_hour))
            ),
            "notification_emails": notification_emails or [],
        }
        terms.update(overrides)
        return cls(**terms)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowance_minutes(self) -> int:
        """Total tolerated delay before a breach, grace included."""
        return self.max_allowable_delay_minutes + self.grace_period_minutes

    def is_active_at(self, when: datetime) -> bool:
        """True when these terms applied at `when` (open-ended bounds are unbounded)."""
        if self.effective_from is not None and when < self.effective_from:
            return False
        if self.effective_to is not None and when >= self.effective_to:
            return False
        return True

    def assess(self, event: DelayEventWebhook, *, now: datetime | None = None) -> SLAAssessment:
        """Deterministically evaluate `event` against these terms.

        Pure arithmetic: no model, no network. Given the same inputs this must always
        produce the same assessment, because it is the evidence a dispute will rest on.

        Penalty accrues on the **excess** beyond allowance + grace only, which is the usual
        contractual reading; `penalty_basis` then decides whether it rounds up per started
        hour or accrues pro-rata per minute, and `penalty_cap` clamps it.
        """
        if event.shipment_id != self.shipment_id:
            raise ValueError(
                f"event {event.event_id} belongs to shipment {event.shipment_id}, "
                f"but this SLA covers {self.shipment_id}"
            )
        assessed_at = now or datetime.now(UTC)
        if not self.is_active_at(assessed_at):
            raise ValueError(f"SLA {self.sla_id} v{self.version} is not in force at {assessed_at}")

        delay = event.effective_delay_minutes
        allowance = self.allowance_minutes
        excess = max(0, delay - allowance)
        breached = excess > 0

        if breached:
            if self.penalty_basis is PenaltyBasis.PER_STARTED_HOUR:
                billable_hours = Decimal(-(-excess // 60))
            else:
                billable_hours = Decimal(excess) / Decimal(60)
            penalty = (billable_hours * self.penalty_per_hour).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            billable_hours = Decimal("0.00")
            penalty = Decimal("0.00")

        capped = self.penalty_cap is not None and penalty > self.penalty_cap
        if capped and self.penalty_cap is not None:
            penalty = self.penalty_cap

        return SLAAssessment(
            shipment_id=self.shipment_id,
            sla_id=self.sla_id,
            sla_version=self.version,
            customer_tier=self.customer_tier,
            delay_minutes=delay,
            reported_delay_minutes=event.reported_delay_minutes,
            eta_derived_delay_minutes=event.derived_delay_minutes,
            delay_discrepancy_minutes=event.delay_discrepancy_minutes,
            allowance_minutes=allowance,
            excess_minutes=excess,
            breached=breached,
            breach_ratio=round(excess / max(allowance, 1), 4),
            penalty_basis=self.penalty_basis,
            billable_hours=billable_hours,
            estimated_penalty=penalty,
            currency=self.currency,
            penalty_capped=capped,
            assessed_at=assessed_at,
        )


class SLAAssessment(BaseModel):
    """Record of the deterministic part of a triage decision."""

    model_config = ConfigDict(extra="forbid")

    shipment_id: ShipmentId
    sla_id: str = Field(max_length=64)
    sla_version: int = Field(ge=1)
    customer_tier: CustomerTier
    delay_minutes: int = Field(ge=0, description="Effective delay used for the assessment.")
    reported_delay_minutes: int = Field(ge=0)
    eta_derived_delay_minutes: int | None = Field(default=None, ge=0)
    delay_discrepancy_minutes: int | None = None
    allowance_minutes: int = Field(ge=0)
    excess_minutes: int = Field(ge=0)
    breached: bool
    breach_ratio: float = Field(ge=0.0, allow_inf_nan=False, description="excess / allowance")
    penalty_basis: PenaltyBasis
    billable_hours: Decimal = Field(ge=0, max_digits=14, decimal_places=2)
    estimated_penalty: Money
    currency: CurrencyCode
    penalty_capped: bool = False
    assessed_at: AwareDatetime

    @model_validator(mode="after")
    def _check_consistency(self) -> SLAAssessment:
        if self.breached != (self.excess_minutes > 0):
            raise ValueError("breached must be exactly equivalent to excess_minutes > 0")
        # Identity: whatever fits inside the allowance, plus whatever spilled over it,
        # must account for the whole delay -- for breaching *and* non-breaching events.
        if min(self.delay_minutes, self.allowance_minutes) + self.excess_minutes != self.delay_minutes:
            raise ValueError(
                "excess_minutes must be exactly the part of delay_minutes beyond "
                f"allowance_minutes ({self.delay_minutes} vs {self.allowance_minutes} + "
                f"{self.excess_minutes})"
            )
        if not self.breached and self.estimated_penalty != 0:
            raise ValueError("a non-breaching delay cannot carry a penalty")
        return self

    @property
    def carrier_understated_delay(self) -> bool:
        """True when the ETA pair implies more delay than the carrier admitted."""
        return (self.delay_discrepancy_minutes or 0) > 0


def severity_from_assessment(assessment: SLAAssessment) -> Severity:
    """Baseline severity from the arithmetic -- the floor an LLM may argue with, not below.

    Graded on how far the delay overshoots the allowance (a 2-hour overshoot on a 15-minute
    allowance is a far worse failure than the same overshoot on a 4-hour one), with a
    one-level bump for VIP customers. Deterministic, so it is unit-testable and a model
    outage still yields a defensible grading.
    """
    if not assessment.breached:
        return Severity.LOW
    if assessment.breach_ratio <= 0.5:
        level = Severity.MEDIUM
    elif assessment.breach_ratio <= 1.5:
        level = Severity.HIGH
    else:
        level = Severity.CRITICAL
    if assessment.customer_tier is CustomerTier.VIP:
        level = _SEVERITY_BUMP[level]
    return level


# --------------------------------------------------------------------------------------
# Outbound contract
# --------------------------------------------------------------------------------------


class EmailDraft(BaseModel):
    """A ready-to-send customer notification.

    Replaces the v1 `email_draft: str`: a bare body string cannot express who receives it
    or what the subject is, which makes automated sending impossible to police.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    channel: NotificationChannel = NotificationChannel.EMAIL
    to: list[EmailAddress] = Field(min_length=1, max_length=20)
    cc: list[EmailAddress] = Field(default_factory=list, max_length=20)
    subject: str = Field(min_length=3, max_length=200)
    body: str = Field(min_length=1, max_length=8_000)
    language: str = Field(default="en", min_length=2, max_length=10)
    generated_by_model: str | None = Field(default=None, max_length=64)

    @field_validator("subject")
    @classmethod
    def _no_header_injection(cls, value: str) -> str:
        if "\r" in value or "\n" in value:
            raise ValueError("subject must be a single line (SMTP header-injection guard)")
        return value

    @field_validator("to", "cc")
    @classmethod
    def _dedupe_recipients(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


def _check_decision_algebra(
    *,
    severity: Severity,
    action: TriageAction,
    email_draft: EmailDraft | None,
    escalation_reason: str | None,
) -> None:
    """The rules that make a routing decision internally consistent.

    Shared by @@TriageProposal@@ (what the model is allowed to ask for) and @@TriageResult@@
    (what the service commits to), so a proposal that would be rejected downstream is
    rejected at the model boundary too -- an operator never has to guess which layer said no.
    """
    if severity is Severity.CRITICAL and action is not TriageAction.HUMAN_ESCALATION:
        raise ValueError("CRITICAL severity must be handled by a human, never auto-sent")
    if action is TriageAction.AUTO_EMAIL:
        if email_draft is None:
            raise ValueError("AUTO_EMAIL requires an email_draft to send")
    elif action is TriageAction.HUMAN_ESCALATION:
        if not escalation_reason or len(escalation_reason) < 3:
            raise ValueError(
                "HUMAN_ESCALATION requires an escalation_reason of at least 3 characters"
            )
    elif action is TriageAction.NO_ACTION:
        if email_draft is not None:
            raise ValueError("NO_ACTION must not carry an e-mail draft")
        if severity in (Severity.HIGH, Severity.CRITICAL):
            raise ValueError(f"NO_ACTION contradicts severity {severity}")


class TriageResult(BaseModel):
    """The decision record: what we concluded, what we will do, and why.

    Validators enforce the decision algebra that the v1 draft left entirely open, so an
    over-eager model cannot emit a record that is internally contradictory:

    * `AUTO_EMAIL` requires a draft and forbids `CRITICAL` severity.
    * `HUMAN_ESCALATION` requires a reason.
    * `NO_ACTION` carries no draft and no `HIGH`/`CRITICAL` severity.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision_id: str = Field(default_factory=lambda: f"trg_{uuid.uuid4().hex}", max_length=64)
    schema_version: str = Field(default=SCHEMA_VERSION, max_length=16)
    event_id: EventId
    shipment_id: ShipmentId
    assessment: SLAAssessment
    severity: Severity
    action: TriageAction
    confidence_score: ConfidenceScore
    email_draft: EmailDraft | None = None
    escalation_reason: str | None = Field(default=None, max_length=1_000)
    reasoning_summary: str = Field(
        min_length=1,
        max_length=2_000,
        description="Human-readable justification, shown to operators.",
    )
    signals: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Short evidence tags that drove the grading.",
    )
    flags: list[TriageFlag] = Field(default_factory=list, max_length=24)
    model_name: str | None = Field(default=None, max_length=64)
    prompt_version: str | None = Field(default=None, max_length=32)
    policy_version: str = Field(default="unapplied", max_length=32)
    latency_ms: int | None = Field(default=None, ge=0)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _enforce_decision_algebra(self) -> TriageResult:
        if self.shipment_id != self.assessment.shipment_id:
            raise ValueError(
                f"result shipment {self.shipment_id} does not match assessment shipment "
                f"{self.assessment.shipment_id}"
            )
        _check_decision_algebra(
            severity=self.severity,
            action=self.action,
            email_draft=self.email_draft,
            escalation_reason=self.escalation_reason,
        )
        return self

    @property
    def will_notify_customer(self) -> bool:
        return self.action is TriageAction.AUTO_EMAIL

    @property
    def email_body(self) -> str | None:
        """Convenience accessor mirroring the v1 `email_draft: str` ergonomics."""
        return None if self.email_draft is None else self.email_draft.body

    @classmethod
    def safe_fallback(
        cls,
        *,
        event: DelayEventWebhook,
        assessment: SLAAssessment,
        reason: str,
        policy_version: str = "unapplied",
        model_name: str | None = None,
        prompt_version: str | None = None,
        latency_ms: int | None = None,
        extra_flags: list[TriageFlag] | None = None,
    ) -> TriageResult:
        """Conservative decision used whenever automated triage fails.

        Called on schema-validation failure of a model response, provider timeouts, or any
        unexpected exception. It never sends anything: the worst outcome of an agent failure
        must be an operator seeing a task, not a customer receiving a bad e-mail.
        """
        flags = [TriageFlag.VALIDATION_FALLBACK, *(extra_flags or [])]
        if event.injection_suspected:
            flags.append(TriageFlag.SUSPECTED_PROMPT_INJECTION)
        return cls(
            event_id=event.event_id,
            shipment_id=event.shipment_id,
            assessment=assessment,
            severity=severity_from_assessment(assessment),
            action=TriageAction.HUMAN_ESCALATION,
            confidence_score=0.0,
            escalation_reason=f"Automated triage unavailable: {reason}"[:1_000],
            reasoning_summary=(
                "Deterministic SLA assessment completed, but no valid automated decision "
                "could be produced, so this exception was routed to a human operator. "
                f"Reason: {reason}"
            )[:2_000],
            flags=list(dict.fromkeys(flags)),
            model_name=model_name,
            prompt_version=prompt_version,
            policy_version=policy_version,
            latency_ms=latency_ms,
        )


class TriageProposal(BaseModel):
    """What the triage model is allowed to decide -- and, deliberately, nothing more.

    This is the schema bound to the LLM's structured output instead of @@TriageResult@@.
    It has no @@assessment@@, no @@estimated_penalty@@ and no identifiers, so a model cannot
    invent a euro figure, mis-attribute a shipment or fabricate a decision id: those are
    merged in from the deterministic @@SLAAssessment@@ *after* the model has spoken.

    The same decision algebra as @@TriageResult@@ applies here, so a nonsensical routing
    request fails at the model boundary and triggers the repair path rather than silently
    becoming an operator's problem.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    severity: Severity = Field(description="Graded urgency. May exceed, never fall below, the computed floor.")
    action: TriageAction
    confidence_score: ConfidenceScore
    email_draft: EmailDraft | None = Field(
        default=None, description="Required when action is AUTO_EMAIL."
    )
    escalation_reason: str | None = Field(default=None, max_length=1_000)
    reasoning_summary: str = Field(min_length=1, max_length=2_000)
    signals: list[str] = Field(default_factory=list, max_length=20)
    flags: list[TriageFlag] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def _enforce_decision_algebra(self) -> TriageProposal:
        _check_decision_algebra(
            severity=self.severity,
            action=self.action,
            email_draft=self.email_draft,
            escalation_reason=self.escalation_reason,
        )
        return self


class TriagePolicy(BaseModel):
    """Guardrails that decide whether an automated decision is allowed to act.

    Policy is data, not code: it is versioned, reviewable by compliance, and stamped onto
    every result it touched. A model may *propose* `AUTO_EMAIL`; this object is what lets
    it through.
    """

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(default="1.0", max_length=32)
    min_confidence_for_auto_email: ConfidenceScore = 0.80
    auto_email_severities: frozenset[Severity] = frozenset({Severity.LOW, Severity.MEDIUM})
    always_escalate_tiers: frozenset[CustomerTier] = frozenset({CustomerTier.VIP})
    max_penalty_for_auto_email: Money = Decimal("250.00")
    max_understatement_for_auto_email: int = Field(default=30, ge=0)

    def apply(self, result: TriageResult) -> TriageResult:
        """Return `result` with policy applied, stamping the policy version either way.

        Downgrades `AUTO_EMAIL` to `HUMAN_ESCALATION` when any guardrail trips, and records
        the specific flags that tripped so operators can see *why* automation stood down
        instead of guessing.
        """
        updates: dict[str, Any] = {"policy_version": self.policy_version}
        if result.action is TriageAction.AUTO_EMAIL:
            blocks: list[TriageFlag] = []
            details: list[str] = []
            assessment = result.assessment
            if result.confidence_score < self.min_confidence_for_auto_email:
                blocks.append(TriageFlag.LOW_CONFIDENCE)
                details.append(
                    f"confidence {result.confidence_score:.2f} is below the "
                    f"{self.min_confidence_for_auto_email:.2f} floor"
                )
            if result.severity not in self.auto_email_severities:
                blocks.append(TriageFlag.SEVERITY_TOO_HIGH_FOR_AUTOMATION)
                details.append(f"severity {result.severity} is not auto-emailable")
            if assessment.customer_tier in self.always_escalate_tiers:
                blocks.append(TriageFlag.VIP_REQUIRES_HUMAN)
                details.append(f"{assessment.customer_tier} customers always get a human")
            if assessment.estimated_penalty > self.max_penalty_for_auto_email:
                blocks.append(TriageFlag.PENALTY_ABOVE_AUTO_LIMIT)
                details.append(
                    f"exposure {assessment.estimated_penalty} {assessment.currency} exceeds the "
                    f"{self.max_penalty_for_auto_email} auto limit"
                )
            if (assessment.delay_discrepancy_minutes or 0) > self.max_understatement_for_auto_email:
                blocks.append(TriageFlag.DELAY_DISCREPANCY)
                details.append(
                    f"carrier under-reported the delay by {assessment.delay_discrepancy_minutes} minutes"
                )
            if blocks:
                reason = (
                    f"Guardrail policy v{self.policy_version} forced human review: "
                    + "; ".join(details)
                    + "."
                )[:1_000]
                prior = result.escalation_reason
                updates.update(
                    action=TriageAction.HUMAN_ESCALATION,
                    escalation_reason=f"{reason} {prior}".strip() if prior else reason,
                    flags=list(dict.fromkeys([*result.flags, *blocks, TriageFlag.POLICY_DOWNGRADE])),
                )
        return revalidated(result, **updates)


class WebhookAck(BaseModel):
    """Immediate response to a carrier webhook: accepted, deduped, queued for triage.

    Triage itself is asynchronous, so this deliberately carries no decision.
    """

    model_config = ConfigDict(extra="forbid")

    accepted: bool = True
    event_id: EventId
    idempotency_key: str = Field(max_length=128)
    duplicate: bool = False
    decision_id: str | None = Field(default=None, max_length=64)
    received_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    message: str = Field(default="queued for asynchronous triage", max_length=200)
