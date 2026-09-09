"""P0-1 (root cause A) regression: the live-camera event-creation schema must
carry the stable actor/object identity fields so `POST /api/events` can
populate the DB's stable-identity columns.

`backend/routers/events.py:create_event` persists via
`models.Event(**event.model_dump(exclude_unset=True))`, so every field declared
on `EventCreate` must map 1:1 onto a `models.Event` column.
"""

from __future__ import annotations

from backend.models import Event
from backend.schemas import EventCreate, EventOut

STABLE_IDENTITY_FIELDS = (
    "event_actor_person_track_id",
    "event_actor_person_uid",
    "event_object_track_id",
    "event_object_uid",
)


def test_event_create_declares_all_four_stable_identity_fields():
    for name in STABLE_IDENTITY_FIELDS:
        assert name in EventCreate.model_fields, (
            f"EventCreate is missing {name}; the live-camera path cannot "
            "populate the DB stable-identity columns without it"
        )


def test_event_create_accepts_and_dumps_stable_identity():
    ev = EventCreate(
        camera_id=1,
        object_type="trash_bag",
        confidence=0.9,
        event_actor_person_track_id=242,
        event_actor_person_uid=2,
        event_object_track_id=10244,
        event_object_uid=100004,
    )
    assert ev.event_actor_person_track_id == 242
    assert ev.event_actor_person_uid == 2
    assert ev.event_object_track_id == 10244
    assert ev.event_object_uid == 100004

    data = ev.model_dump(exclude_unset=True)
    for name in STABLE_IDENTITY_FIELDS:
        assert name in data
    # every dumped identity field must be a real Event column so that
    # models.Event(**data) in create_event keeps working
    for name in STABLE_IDENTITY_FIELDS:
        assert hasattr(Event, name)


def test_event_create_identity_fields_default_to_none():
    ev = EventCreate(camera_id=1, object_type="trash_bag")
    for name in STABLE_IDENTITY_FIELDS:
        assert getattr(ev, name) is None


def test_event_out_exposes_stable_identity_fields():
    for name in STABLE_IDENTITY_FIELDS:
        assert name in EventOut.model_fields
