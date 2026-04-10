from __future__ import annotations

import pytest

from rl_task_foundry.config.models import DomainConfig, ModelRef, ProviderConfig
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.schema.path_catalog import build_path_catalog
from rl_task_foundry.tools.compiler import compile_path_tools
from rl_task_foundry.tools.model_naming import (
    ToolNameVariant,
    ToolNameVariantResponse,
    ToolNamingGenerationError,
    _naming_prompt,
    _parse_tool_name_variant_response,
    generate_named_tool_bundle,
)


def _synthetic_bundle():
    graph = SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="orders",
                primary_key=("order_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="order_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="shipment_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                ],
            ),
            TableProfile(
                schema_name="public",
                table_name="shipments",
                primary_key=("shipment_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="shipment_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="amount",
                        data_type="numeric",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_shipments_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("shipment_id",),
                target_schema="public",
                target_table="shipments",
                target_columns=("shipment_id",),
            )
        ],
    )
    path = build_path_catalog(graph, max_hops=2).get("orders.shipments")
    bundle = compile_path_tools(graph, path, tool_level=2, max_list_cardinality=5)
    return path, bundle


@pytest.mark.asyncio
async def test_generate_named_tool_bundle_marks_model_generated(monkeypatch):
    path, bundle = _synthetic_bundle()

    async def _fake_request(**_kwargs):
        return ToolNameVariantResponse(
            variants=[
                ToolNameVariant(
                    semantic_key=tool.semantic_key,
                    name=f"tool_{index}_{tool.kind}",
                    description=f"generated {tool.kind}",
                )
                for index, tool in enumerate(bundle.tools, start=1)
            ]
        )

    monkeypatch.setattr(
        "rl_task_foundry.tools.model_naming._request_tool_name_variants",
        _fake_request,
    )

    renamed = await generate_named_tool_bundle(
        provider=ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:10531/v1",
            api_key_env="OPENAI_API_KEY",
        ),
        model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
        domain=DomainConfig(name="customer_support", language="ko"),
        path=path,
        bundle=bundle,
    )

    assert {tool.name_source for tool in renamed.tools} == {"model_generated"}
    assert [tool.semantic_key for tool in renamed.tools] == [tool.semantic_key for tool in bundle.tools]
    assert renamed.tools[0].name.startswith("tool_")


@pytest.mark.asyncio
async def test_generate_named_tool_bundle_rejects_duplicate_names(monkeypatch):
    path, bundle = _synthetic_bundle()

    async def _fake_request(**_kwargs):
        return ToolNameVariantResponse(
            variants=[
                ToolNameVariant(
                    semantic_key=tool.semantic_key,
                    name="duplicate_name",
                    description="generated",
                )
                for tool in bundle.tools
            ]
        )

    monkeypatch.setattr(
        "rl_task_foundry.tools.model_naming._request_tool_name_variants",
        _fake_request,
    )

    with pytest.raises(ToolNamingGenerationError, match="failed after 3 attempts") as exc_info:
        await generate_named_tool_bundle(
            provider=ProviderConfig(
                type="openai_compatible",
                base_url="http://127.0.0.1:10531/v1",
                api_key_env="OPENAI_API_KEY",
            ),
            model_ref=ModelRef(provider="codex_oauth", model="gpt-5.4-mini"),
            domain=DomainConfig(name="customer_support", language="ko"),
            path=path,
            bundle=bundle,
        )
    assert exc_info.value.__cause__ is not None
    assert "duplicate" in str(exc_info.value.__cause__)


def test_parse_tool_name_variant_response_accepts_tools_alias():
    parsed = _parse_tool_name_variant_response(
        """
        {
          "tools": [
            {
              "semantic_key": "customer.address.city:lookup",
              "name": "browse_customer_geo_city_rows",
              "description": "L2 lookup for reachable city rows."
            }
          ]
        }
        """
    )

    assert parsed.variants[0].semantic_key == "customer.address.city:lookup"


def test_parse_tool_name_variant_response_ignores_extra_fields():
    parsed = _parse_tool_name_variant_response(
        """
        {
          "variants": [
            {
              "semantic_key": "customer.store.staff:lookup",
              "name": "lookup_service_contact_owner",
              "description": "Find the staff contact for the customer context.",
              "kind": "lookup",
              "current_name": "get_staff_for_customer_via_store",
              "parameters": ["anchor_customer_id"],
              "output_fields": ["first_name", "last_name"]
            }
          ]
        }
        """
    )

    assert parsed.variants[0].name == "lookup_service_contact_owner"


def test_naming_prompt_uses_domain_neutral_examples_for_l2():
    path, bundle = _synthetic_bundle()

    prompt = _naming_prompt(
        domain=DomainConfig(name="logistics_support", language="ko"),
        path=path,
        tool_level=2,
        bundle=bundle,
        task_question="최근 배송 상태를 확인해줘",
        question_family="status_lookup",
        outcome_type="answer",
    )

    content = prompt[1]["content"]
    assert "inspect_customer_address_city_records" not in content
    assert "count_payment_city_country_rows" not in content
    assert '"better": "inspect_profile"' in content
    assert '"better": "review_timeline"' in content
