from rl_task_foundry.infra.visibility import (
    blocks_direct_label_exposure,
    is_blocked_visibility,
    is_user_visible_visibility,
    is_visibility,
    redact_dict,
    resolve_visibility,
)


def test_visibility_resolve_uses_only_overrides_and_default_visibility():
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


def test_visibility_predicates_are_policy_metadata_checks():
    assert is_visibility("blocked")
    assert is_visibility("internal")
    assert is_visibility("user_visible")
    assert not is_visibility("derived")

    assert is_blocked_visibility("blocked")
    assert is_user_visible_visibility("user_visible")
    assert blocks_direct_label_exposure("blocked")
    assert blocks_direct_label_exposure("internal")
    assert not blocks_direct_label_exposure("user_visible")
    assert not blocks_direct_label_exposure("derived")
