import json

from rl_task_foundry.tasks.models import SolverResult, TaskSpec
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema, GroundTruth
from rl_task_foundry.verification.policies import VerificationPolicy
from rl_task_foundry.verification.scorer import VerificationEngine


def test_verification_engine_canonicalizes_and_scores_exact_match():
    schema = AnswerSchema(
        fields=[
            AnswerField(
                name="delivery_status",
                type="string",
                canonicalizer="lower_trim",
            )
        ]
    )
    task = TaskSpec(
        task_id="task_1",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question="현재 배송 상태는 무엇인가요?",
        answer_schema=schema,
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        sensitivity_policy="default",
    )
    truth = GroundTruth(
        task_id="task_1",
        verification_sql="SELECT 1",
        canonical_answer={"delivery_status": "in_transit"},
        answer_schema_version="v1",
    )
    result = SolverResult(
        task_id="task_1",
        solver_id="solver_a",
        provider="codex_oauth",
        model="gpt-5.4-mini",
        replica_index=0,
        transcript_ref="trace://transcript",
        tool_trace_ref="trace://tools",
        raw_output_text="IN_TRANSIT",
        structured_output={"delivery_status": "IN_TRANSIT"},
        status="completed",
    )

    verify = VerificationEngine().verify(task, truth, result)
    assert verify.pass_exact is True
    assert verify.field_scores == {"delivery_status": True}


def test_verification_engine_preserves_submission_failure_reason():
    schema = AnswerSchema(
        fields=[
            AnswerField(
                name="delivery_status",
                type="string",
                canonicalizer="lower_trim",
            )
        ]
    )
    task = TaskSpec(
        task_id="task_1",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question="현재 배송 상태는 무엇인가요?",
        answer_schema=schema,
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        sensitivity_policy="default",
    )
    truth = GroundTruth(
        task_id="task_1",
        verification_sql="SELECT 1",
        canonical_answer={"delivery_status": "in_transit"},
        answer_schema_version="v1",
    )
    result = SolverResult(
        task_id="task_1",
        solver_id="solver_a",
        provider="codex_oauth",
        model="gpt-5.4-mini",
        replica_index=0,
        transcript_ref="trace://transcript",
        tool_trace_ref="trace://tools",
        raw_output_text='{"submitted": false}',
        structured_output=None,
        status="invalid_submit",
        termination_reason="invalid_submit_schema",
        termination_metadata={"error": "submit_result payload failed schema validation"},
    )

    verify = VerificationEngine().verify(task, truth, result)
    assert verify.pass_exact is False
    assert verify.failure_reason == "invalid_submit_schema"


def test_verification_engine_fails_on_internal_field_leak():
    schema = AnswerSchema(
        fields=[
            AnswerField(
                name="delivery_status",
                type="string",
                canonicalizer="lower_trim",
                visibility="user_visible",
                source_columns=["shipments.status"],
            ),
            AnswerField(
                name="courier_phone",
                type="string",
                canonicalizer="lower_trim",
                visibility="internal",
                source_columns=["shipments.phone"],
            ),
        ]
    )
    task = TaskSpec(
        task_id="task_1",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question="현재 배송 상태는 무엇인가요?",
        answer_schema=schema,
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        sensitivity_policy="default",
    )
    truth = GroundTruth(
        task_id="task_1",
        verification_sql="SELECT 1",
        canonical_answer={"delivery_status": "in_transit"},
        answer_schema_version="v1",
    )
    result = SolverResult(
        task_id="task_1",
        solver_id="solver_a",
        provider="codex_oauth",
        model="gpt-5.4-mini",
        replica_index=0,
        transcript_ref="trace://transcript",
        tool_trace_ref="trace://tools",
        raw_output_text='{"delivery_status":"IN_TRANSIT","courier_phone":"010-1111-2222"}',
        structured_output={
            "delivery_status": "IN_TRANSIT",
            "courier_phone": "010-1111-2222",
        },
        status="completed",
    )

    verify = VerificationEngine().verify(task, truth, result)
    assert verify.field_scores == {"delivery_status": True}
    assert verify.pass_exact is False
    assert verify.failure_reason == "internal_field_leak"


def test_verification_engine_can_allow_internal_field_for_diagnostics():
    schema = AnswerSchema(
        fields=[
            AnswerField(
                name="delivery_status",
                type="string",
                canonicalizer="lower_trim",
                visibility="user_visible",
            ),
            AnswerField(
                name="courier_phone",
                type="string",
                canonicalizer="lower_trim",
                visibility="internal",
            ),
        ]
    )
    task = TaskSpec(
        task_id="task_1",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question="현재 배송 상태는 무엇인가요?",
        answer_schema=schema,
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        sensitivity_policy="default",
    )
    truth = GroundTruth(
        task_id="task_1",
        verification_sql="SELECT 1",
        canonical_answer={"delivery_status": "in_transit"},
        answer_schema_version="v1",
    )
    result = SolverResult(
        task_id="task_1",
        solver_id="solver_a",
        provider="codex_oauth",
        model="gpt-5.4-mini",
        replica_index=0,
        transcript_ref="trace://transcript",
        tool_trace_ref="trace://tools",
        raw_output_text='{"delivery_status":"IN_TRANSIT","courier_phone":"010-1111-2222"}',
        structured_output={
            "delivery_status": "IN_TRANSIT",
            "courier_phone": "010-1111-2222",
        },
        status="completed",
    )

    verify = VerificationEngine(
        policy=VerificationPolicy(require_provenance=True, fail_on_internal_field_leak=False)
    ).verify(task, truth, result)
    assert verify.pass_exact is True


def test_verification_engine_uses_configured_float_precision():
    schema = AnswerSchema(
        fields=[
            AnswerField(
                name="avg_amount",
                type="float",
                canonicalizer="round_custom",
            )
        ]
    )
    task = TaskSpec(
        task_id="task_float",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="B",
        question="평균 금액은 얼마인가요?",
        answer_schema=schema,
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        sensitivity_policy="default",
    )
    truth = GroundTruth(
        task_id="task_float",
        verification_sql="SELECT 1",
        canonical_answer={"avg_amount": 12.346},
        answer_schema_version="v1",
    )
    result = SolverResult(
        task_id="task_float",
        solver_id="solver_a",
        provider="codex_oauth",
        model="gpt-5.4-mini",
        replica_index=0,
        transcript_ref="trace://transcript",
        tool_trace_ref="trace://tools",
        raw_output_text='{"avg_amount":12.3456}',
        structured_output={"avg_amount": 12.3456},
        status="completed",
    )

    verify = VerificationEngine(
        policy=VerificationPolicy(float_precision=3)
    ).verify(task, truth, result)

    assert verify.pass_exact is True
    assert verify.field_scores == {"avg_amount": True}
    assert verify.failure_reason is None


def test_verification_engine_runs_shadow_verifier_and_marks_match():
    schema = AnswerSchema(
        fields=[
            AnswerField(
                name="delivery_status",
                type="string",
                canonicalizer="lower_trim",
                source_columns=["shipments.status"],
            )
        ]
    )
    task = TaskSpec(
        task_id="task_shadow_match",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question="현재 배송 상태는 무엇인가요?",
        answer_schema=schema,
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        sensitivity_policy="default",
    )
    truth = GroundTruth(
        task_id="task_shadow_match",
        verification_sql="SELECT 1",
        canonical_answer={"delivery_status": "in_transit"},
        row_context=[{"delivery_status": "in_transit"}],
        answer_schema_version="v1",
    )
    result = SolverResult(
        task_id="task_shadow_match",
        solver_id="solver_a",
        provider="codex_oauth",
        model="gpt-5.4-mini",
        replica_index=0,
        transcript_ref="trace://transcript",
        tool_trace_ref="trace://tools",
        raw_output_text='{"delivery_status":"IN_TRANSIT"}',
        structured_output={"delivery_status": "IN_TRANSIT"},
        status="completed",
    )

    verify = VerificationEngine(
        policy=VerificationPolicy(shadow_sample_rate=1.0)
    ).verify(task, truth, result)

    assert verify.pass_exact is True
    assert verify.shadow_verifier_status == "match"
    assert verify.shadow_pass_exact is True
    assert verify.shadow_failure_reason is None


def test_verification_engine_marks_shadow_disagreement_when_row_context_differs():
    schema = AnswerSchema(
        fields=[
            AnswerField(
                name="delivery_status",
                type="string",
                canonicalizer="lower_trim",
                source_columns=["shipments.status"],
            )
        ]
    )
    task = TaskSpec(
        task_id="task_shadow_disagree",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question="현재 배송 상태는 무엇인가요?",
        answer_schema=schema,
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        sensitivity_policy="default",
    )
    truth = GroundTruth(
        task_id="task_shadow_disagree",
        verification_sql="SELECT 1",
        canonical_answer={"delivery_status": "delivered"},
        row_context=[{"delivery_status": "in_transit"}],
        answer_schema_version="v1",
    )
    result = SolverResult(
        task_id="task_shadow_disagree",
        solver_id="solver_a",
        provider="codex_oauth",
        model="gpt-5.4-mini",
        replica_index=0,
        transcript_ref="trace://transcript",
        tool_trace_ref="trace://tools",
        raw_output_text='{"delivery_status":"IN_TRANSIT"}',
        structured_output={"delivery_status": "IN_TRANSIT"},
        status="completed",
    )

    verify = VerificationEngine(
        policy=VerificationPolicy(shadow_sample_rate=1.0)
    ).verify(task, truth, result)

    assert verify.pass_exact is False
    assert verify.failure_reason == "field_mismatch"
    assert verify.shadow_verifier_status == "disagree"
    assert verify.shadow_pass_exact is True
    assert verify.shadow_failure_reason is None


def test_verification_engine_fails_when_required_core_tool_not_used(tmp_path):
    schema = AnswerSchema(
        fields=[
            AnswerField(
                name="delivery_status",
                type="string",
                canonicalizer="lower_trim",
                visibility="user_visible",
                source_columns=["shipments.status"],
            )
        ]
    )
    tool_trace_path = tmp_path / "tool_trace.json"
    tool_trace_path.write_text(
        json.dumps(
            {
                "run_items": [
                    "'tool-call(other_lookup)'",
                    "'tool-call(submit_result)'",
                ],
                "tool_calls": [
                    {"name": "other_lookup", "repr": "'tool-call(other_lookup)'"},
                    {"name": "submit_result", "repr": "'tool-call(submit_result)'"},
                ],
            }
        ),
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="task_1",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question="현재 배송 상태는 무엇인가요?",
        answer_schema=schema,
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        provenance_requirements=["delivery_lookup"],
        sensitivity_policy="default",
    )
    truth = GroundTruth(
        task_id="task_1",
        verification_sql="SELECT 1",
        canonical_answer={"delivery_status": "in_transit"},
        answer_schema_version="v1",
    )
    result = SolverResult(
        task_id="task_1",
        solver_id="solver_a",
        provider="codex_oauth",
        model="gpt-5.4-mini",
        replica_index=0,
        transcript_ref="trace://transcript",
        tool_trace_ref=str(tool_trace_path),
        raw_output_text="IN_TRANSIT",
        structured_output={"delivery_status": "IN_TRANSIT"},
        status="completed",
    )

    verify = VerificationEngine().verify(task, truth, result)
    assert verify.pass_exact is False
    assert verify.provenance_pass is False


def test_verification_engine_passes_when_required_core_tool_used(tmp_path):
    schema = AnswerSchema(
        fields=[
            AnswerField(
                name="delivery_status",
                type="string",
                canonicalizer="lower_trim",
                visibility="user_visible",
                source_columns=["shipments.status"],
            )
        ]
    )
    tool_trace_path = tmp_path / "tool_trace.json"
    tool_trace_path.write_text(
        json.dumps(
            {
                "run_items": [
                    "'tool-call(delivery_lookup)'",
                    "'tool-call(submit_result)'",
                ],
                "tool_calls": [
                    {"name": "delivery_lookup", "repr": "'tool-call(delivery_lookup)'"},
                    {"name": "submit_result", "repr": "'tool-call(submit_result)'"},
                ],
            }
        ),
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="task_1",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question="현재 배송 상태는 무엇인가요?",
        answer_schema=schema,
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        provenance_requirements=["delivery_lookup"],
        sensitivity_policy="default",
    )
    truth = GroundTruth(
        task_id="task_1",
        verification_sql="SELECT 1",
        canonical_answer={"delivery_status": "in_transit"},
        answer_schema_version="v1",
    )
    result = SolverResult(
        task_id="task_1",
        solver_id="solver_a",
        provider="codex_oauth",
        model="gpt-5.4-mini",
        replica_index=0,
        transcript_ref="trace://transcript",
        tool_trace_ref=str(tool_trace_path),
        raw_output_text="IN_TRANSIT",
        structured_output={"delivery_status": "IN_TRANSIT"},
        status="completed",
    )

    verify = VerificationEngine().verify(task, truth, result)
    assert verify.pass_exact is True
    assert verify.provenance_pass is True
