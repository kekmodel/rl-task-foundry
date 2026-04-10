from __future__ import annotations

from datetime import date, datetime

from rl_task_foundry.truth.canonicalize import canonicalize_answer
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema


def test_canonicalize_answer_normalizes_date_datetime_bool_and_float() -> None:
    schema = AnswerSchema(
        fields=[
            AnswerField(name="ship_date", type="date", canonicalizer="iso_date"),
            AnswerField(name="shipped_at", type="datetime", canonicalizer="iso_datetime"),
            AnswerField(name="is_active", type="bool", canonicalizer="identity"),
            AnswerField(name="avg_amount", type="float", canonicalizer="round_6"),
        ]
    )

    canonical = canonicalize_answer(
        schema,
        {
            "ship_date": datetime(2026, 4, 11, 15, 30, 0),
            "shipped_at": "2026-04-11T15:30:45",
            "is_active": "true",
            "avg_amount": 12.34567891,
        },
    )

    assert canonical == {
        "ship_date": "2026-04-11",
        "shipped_at": "2026-04-11T15:30:45",
        "is_active": True,
        "avg_amount": 12.345679,
    }


def test_canonicalize_answer_normalizes_date_object() -> None:
    schema = AnswerSchema(
        fields=[AnswerField(name="ship_date", type="date", canonicalizer="iso_date")]
    )

    canonical = canonicalize_answer(schema, {"ship_date": date(2026, 4, 11)})

    assert canonical == {"ship_date": "2026-04-11"}


def test_canonicalize_answer_supports_custom_float_precision() -> None:
    schema = AnswerSchema(
        fields=[AnswerField(name="avg_amount", type="float", canonicalizer="round_custom")]
    )

    canonical = canonicalize_answer(
        schema,
        {"avg_amount": 12.34567891},
        float_precision=3,
    )

    assert canonical == {"avg_amount": 12.346}
