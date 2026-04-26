from __future__ import annotations

from rl_task_foundry.schema.graph import (
    ColumnProfile,
    ForeignKeyEdge,
    SchemaGraph,
    TableProfile,
)
from rl_task_foundry.schema.profiler import (
    CategoricalColumnStats,
    DataProfile,
    NumericColumnStats,
)
from rl_task_foundry.synthesis.affordance_map import build_db_affordance_map


def _column(
    table: str,
    name: str,
    data_type: str = "text",
    *,
    ordinal: int = 1,
    visibility: str = "user_visible",
    primary_key: bool = False,
    foreign_key: bool = False,
) -> ColumnProfile:
    return ColumnProfile(
        schema_name="public",
        table_name=table,
        column_name=name,
        data_type=data_type,
        ordinal_position=ordinal,
        is_nullable=False,
        visibility=visibility,  # type: ignore[arg-type]
        is_primary_key=primary_key,
        is_foreign_key=foreign_key,
    )


def _table(
    name: str,
    columns: list[ColumnProfile],
    *,
    rows: int = 100,
    pk: tuple[str, ...] = (),
) -> TableProfile:
    return TableProfile(
        schema_name="public",
        table_name=name,
        columns=columns,
        primary_key=pk,
        row_estimate=rows,
    )


def test_build_db_affordance_map_summarizes_tables_paths_and_topics() -> None:
    customer = _table(
        "customer",
        [
            _column(
                "customer",
                "customer_id",
                "integer",
                visibility="blocked",
                primary_key=True,
            ),
            _column("customer", "display_name", ordinal=2),
            _column("customer", "email", ordinal=3, visibility="internal"),
            _column("customer", "active", "integer", ordinal=4),
            _column("customer", "create_date", "date", ordinal=5),
        ],
        rows=100,
        pk=("customer_id",),
    )
    rental = _table(
        "rental",
        [
            _column(
                "rental",
                "rental_id",
                "integer",
                visibility="blocked",
                primary_key=True,
            ),
            _column(
                "rental",
                "customer_id",
                "integer",
                ordinal=2,
                visibility="blocked",
                foreign_key=True,
            ),
            _column("rental", "rental_date", "timestamp", ordinal=3),
            _column("rental", "status", ordinal=4),
            _column("rental", "update_ts", "timestamp", ordinal=5),
        ],
        rows=800,
        pk=("rental_id",),
    )
    payment = _table(
        "payment",
        [
            _column(
                "payment",
                "payment_id",
                "integer",
                visibility="blocked",
                primary_key=True,
            ),
            _column(
                "payment",
                "customer_id",
                "integer",
                ordinal=2,
                visibility="blocked",
                foreign_key=True,
            ),
            _column("payment", "amount", "numeric", ordinal=3),
            _column("payment", "paid_at", "timestamp", ordinal=4),
            _column("payment", "warning", "integer", ordinal=5),
            _column("payment", "valuenum", "numeric", ordinal=6),
            _column("payment", "linkorderid", "integer", ordinal=7),
            _column("payment", "isopenbag", "integer", ordinal=8),
            _column("payment", "continueinnextdept", "integer", ordinal=9),
            _column("payment", "last_update", "timestamp", ordinal=10),
        ],
        rows=1200,
        pk=("payment_id",),
    )
    graph = SchemaGraph(
        tables=[customer, rental, payment],
        edges=[
            ForeignKeyEdge(
                constraint_name="rental_customer_fk",
                source_schema="public",
                source_table="rental",
                source_columns=("customer_id",),
                target_schema="public",
                target_table="customer",
                target_columns=("customer_id",),
                fanout_estimate=8.0,
            ),
            ForeignKeyEdge(
                constraint_name="payment_customer_fk",
                source_schema="public",
                source_table="payment",
                source_columns=("customer_id",),
                target_schema="public",
                target_table="customer",
                target_columns=("customer_id",),
                fanout_estimate=12.0,
            ),
        ],
    )
    profile = DataProfile(
        numeric=[
            NumericColumnStats(
                table="public.payment",
                column="amount",
                mean=4.5,
                std=1.0,
                min_val=1.0,
                max_val=9.0,
                distinct=20,
                low_threshold=3.5,
                high_threshold=5.5,
            )
        ],
        categorical=[
            CategoricalColumnStats(
                table="public.customer",
                column="email",
                categories=["a@example.test", "b@example.test"],
                counts=[1, 1],
            ),
            CategoricalColumnStats(
                table="public.rental",
                column="status",
                categories=["open", "closed"],
                counts=[500, 300],
            ),
        ],
    )

    affordance_map = build_db_affordance_map(graph, data_profile=profile)
    assert len(affordance_map["table_affordances"]) == 3
    assert len(affordance_map["path_affordances"]) == 2
    assert [
        card["table"]
        for card in affordance_map["table_affordances"]  # type: ignore[index]
    ] == ["public.customer", "public.rental", "public.payment"]
    assert [
        card["path"]
        for card in affordance_map["path_affordances"]  # type: ignore[index]
    ] == [
        "public.customer -> public.rental via rental.customer_id = customer.customer_id",
        "public.customer -> public.payment via payment.customer_id = customer.customer_id",
    ]

    table_cards = {
        str(card["table"]): card
        for card in affordance_map["table_affordances"]  # type: ignore[index]
    }
    assert table_cards["public.customer"]["structure"] == "hub"
    assert "anchor_candidate" in table_cards["public.customer"]["affordances"]
    assert "email" not in table_cards["public.customer"]["categorical_filters"]
    assert "email" not in table_cards["public.customer"]["readable"]
    assert "active" in table_cards["public.customer"]["numeric_metrics"]
    assert "active" not in table_cards["public.customer"]["categorical_filters"]
    assert "create_date" in table_cards["public.customer"]["time_columns"]
    assert table_cards["public.payment"]["structure"] == "referrer"
    assert table_cards["public.payment"]["numeric_metrics"] == [
        "amount",
        "continueinnextdept",
        "isopenbag",
        "linkorderid",
        "valuenum",
    ]
    assert "last_update" in table_cards["public.payment"]["time_columns"]

    path_cards = {
        str(card["path"]): card
        for card in affordance_map["path_affordances"]  # type: ignore[index]
    }
    rental_path = path_cards[
        "public.customer -> public.rental via rental.customer_id = customer.customer_id"
    ]
    assert rental_path["relation"] == "rental.customer_id = customer.customer_id"
    assert rental_path["fanout"] == 8.0
    assert "ordered_list" in rental_path["supports"]
    assert "count_or_cardinality" in rental_path["supports"]
    assert "timeline_or_time_filter" in rental_path["supports"]
    assert "status" in rental_path["filters"]
    assert "update_ts" in rental_path["filters"]

    assert "topic_affordances" not in affordance_map
    assert "highlighted_table_affordances" not in affordance_map
    assert "highlighted_path_affordances" not in affordance_map
