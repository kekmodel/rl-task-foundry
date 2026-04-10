from rl_task_foundry.infra.privacy import infer_visibility, redact_dict, resolve_visibility


def test_privacy_infers_visibility_from_column_names():
    assert infer_visibility("customer_email") == "internal"
    assert infer_visibility("card_number") == "blocked"
    assert infer_visibility("delivery_status") is None


def test_privacy_resolve_and_redact_respects_overrides():
    visibility = resolve_visibility(
        "courier_phone",
        default_visibility="blocked",
        overrides={"courier_phone": "user_visible"},
    )
    assert visibility == "user_visible"

    payload = redact_dict(
        {
            "courier_phone": "010-0000-0000",
            "customer_email": "user@example.com",
            "card_number": "4111111111111111",
        },
        {
            "courier_phone": "user_visible",
            "customer_email": "internal",
            "card_number": "blocked",
        },
    )
    assert payload["courier_phone"] == "010-0000-0000"
    assert payload["customer_email"] == "[INTERNAL]"
    assert payload["card_number"] == "[REDACTED]"
