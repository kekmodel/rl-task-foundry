from rl_task_foundry.infra.privacy import infer_visibility, redact_dict, resolve_visibility


def test_privacy_does_not_infer_visibility_from_column_names():
    assert infer_visibility("customer_email") is None
    assert infer_visibility("card_number") is None
    assert infer_visibility("delivery_status") is None


def test_privacy_resolve_uses_only_overrides_and_default_visibility():
    visibility = resolve_visibility(
        "courier_phone",
        default_visibility="blocked",
        overrides={"courier_phone": "user_visible"},
    )
    assert visibility == "user_visible"
    assert (
        resolve_visibility(
            "card_number",
            default_visibility="user_visible",
            overrides={},
        )
        == "user_visible"
    )

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
