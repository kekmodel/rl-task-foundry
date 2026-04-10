"""Tests for the OpenHarness abstraction layer."""

from __future__ import annotations

from rlvr_synth.abstraction import AgentRunner, ToolDef, ToolResult


def test_tool_def_to_schema() -> None:
    tool = ToolDef(
        name="get_customer",
        description="Get customer by ID",
        parameters={"customer_id": {"type": "integer", "description": "Customer ID"}},
    )
    schema = tool.to_api_schema()
    assert schema["name"] == "get_customer"
    assert "customer_id" in schema["input_schema"]["properties"]


def test_tool_result_creation() -> None:
    result = ToolResult(output="hello", is_error=False)
    assert result.output == "hello"
    assert not result.is_error

    err = ToolResult(output="oops", is_error=True)
    assert err.is_error
