"""Small text helpers shared across tool compilation and evaluation."""

from __future__ import annotations

_PEOPLE_LIKE_TOKENS = {
    "staff",
    "employee",
    "agent",
    "user",
    "customer",
    "member",
    "person",
    "patient",
    "student",
    "teacher",
    "driver",
    "courier",
}
_CASE_LIKE_TOKENS = {
    "order",
    "payment",
    "rental",
    "booking",
    "shipment",
    "request",
    "ticket",
    "case",
    "invoice",
    "transaction",
}
_PLACE_LIKE_TOKENS = {
    "store",
    "branch",
    "site",
    "location",
    "outlet",
    "shop",
    "warehouse",
    "office",
    "station",
}
_KO_ENTITY_LABELS = {
    "address": "주소",
    "booking": "예약",
    "branch": "지점",
    "case": "건",
    "category": "분류",
    "city": "도시",
    "country": "국가",
    "customer": "고객",
    "employee": "직원",
    "inventory": "재고 항목",
    "invoice": "청구서",
    "language": "언어",
    "location": "위치",
    "member": "회원",
    "order": "주문",
    "payment": "결제",
    "person": "사람",
    "profile": "프로필",
    "rental": "대여",
    "request": "요청",
    "shipment": "배송",
    "staff": "직원",
    "store": "매장",
    "student": "학생",
    "teacher": "교사",
    "ticket": "티켓",
    "transaction": "거래",
    "user": "사용자",
}
_EN_ENTITY_LABELS = {
    "address": "address",
    "booking": "booking",
    "branch": "branch",
    "case": "case",
    "category": "category",
    "city": "city",
    "country": "country",
    "customer": "customer",
    "employee": "employee",
    "inventory": "inventory item",
    "invoice": "invoice",
    "language": "language",
    "location": "location",
    "member": "member",
    "order": "order",
    "payment": "payment",
    "person": "person",
    "profile": "profile",
    "rental": "rental",
    "request": "request",
    "shipment": "shipment",
    "staff": "staff member",
    "store": "store",
    "student": "student",
    "teacher": "teacher",
    "ticket": "ticket",
    "transaction": "transaction",
    "user": "user",
}
_KO_COUNT_PHRASES = {
    "booking": "예약 내역",
    "invoice": "청구 내역",
    "order": "주문 내역",
    "payment": "결제 내역",
    "rental": "대여 내역",
    "request": "요청 내역",
    "shipment": "배송 내역",
    "transaction": "거래 내역",
}
_EN_COUNT_PHRASES = {
    "booking": "booking history",
    "invoice": "invoice history",
    "order": "order history",
    "payment": "payment history",
    "rental": "rental history",
    "request": "request history",
    "shipment": "shipment history",
    "transaction": "transaction history",
}


def singularize_token(token: str) -> str:
    """Return a light-weight English singularization for identifier tokens."""

    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 1:
        return token[:-1]
    return token


def humanize_identifier(identifier: str) -> str:
    """Convert a snake_case-ish identifier into a space-delimited label."""

    parts = [part for part in identifier.replace("-", "_").split("_") if part]
    if not parts:
        return identifier.strip()
    return " ".join(parts)


def count_unit_hint_for_identifier(identifier: str) -> str:
    """Infer a coarse count unit for an entity identifier."""

    token = singularize_token(identifier.strip().lower())
    if token in _PEOPLE_LIKE_TOKENS:
        return "people"
    if token in _CASE_LIKE_TOKENS:
        return "cases"
    if token in _PLACE_LIKE_TOKENS:
        return "places"
    return "items"


def localized_entity_label(identifier: str, *, language: str) -> str:
    token = singularize_token(identifier.strip().lower())
    if language == "ko":
        return _KO_ENTITY_LABELS.get(token, humanize_identifier(token))
    return _EN_ENTITY_LABELS.get(token, humanize_identifier(token))


def default_count_target_label(identifier: str, *, language: str) -> str:
    """Return a user-facing target label for count questions."""

    return localized_entity_label(identifier, language=language)


def count_phrase_reference(identifier: str, *, language: str = "en") -> str:
    """Return a neutral concept phrase for aggregate question composition."""

    token = singularize_token(identifier.strip().lower())
    if language == "ko":
        if token in _KO_COUNT_PHRASES:
            return _KO_COUNT_PHRASES[token]
    else:
        if token in _EN_COUNT_PHRASES:
            return _EN_COUNT_PHRASES[token]

    label = localized_entity_label(token, language=language)
    unit = count_unit_hint_for_identifier(token)
    if language == "ko":
        if unit == "people":
            return f"{label} 수"
        if unit == "cases":
            return f"{label} 건수"
        if unit == "places":
            return f"{label} 수"
        return label
    if unit == "people":
        return f"{label} count"
    if unit == "cases":
        return f"{label} count"
    if unit == "places":
        return f"{label} count"
    return label
