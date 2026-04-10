from rl_task_foundry.tasks.models import SolverResult, TaskSpec, VerifyResult
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema, GroundTruth
from rl_task_foundry.verification.shadow import run_shadow_verifier, should_sample_shadow_verifier


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="task_shadow",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question="현재 배송 상태는 무엇인가요?",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="delivery_status",
                    type="string",
                    canonicalizer="lower_trim",
                    source_columns=["shipments.status"],
                )
            ]
        ),
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        sensitivity_policy="default",
    )


def _solver_result() -> SolverResult:
    return SolverResult(
        task_id="task_shadow",
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


def test_should_sample_shadow_verifier_is_deterministic():
    sampled_once = should_sample_shadow_verifier(
        task_id="task_shadow",
        solver_id="solver_a",
        sample_rate=0.5,
    )
    sampled_twice = should_sample_shadow_verifier(
        task_id="task_shadow",
        solver_id="solver_a",
        sample_rate=0.5,
    )
    assert sampled_once is sampled_twice


def test_run_shadow_verifier_reconstructs_from_row_context():
    task = _task()
    truth = GroundTruth(
        task_id="task_shadow",
        verification_sql="SELECT 1",
        canonical_answer={"delivery_status": "delivered"},
        row_context=[{"delivery_status": "in_transit"}],
        answer_schema_version="v1",
    )
    primary = VerifyResult(
        task_id="task_shadow",
        solver_id="solver_a",
        pass_exact=False,
        failure_reason="field_mismatch",
    )

    outcome = run_shadow_verifier(
        task=task,
        ground_truth=truth,
        solver_result=_solver_result(),
        primary_verification=primary,
        provenance_pass=True,
        fail_on_internal_field_leak=True,
        float_precision=6,
    )

    assert outcome.status == "disagree"
    assert outcome.pass_exact is True
    assert outcome.failure_reason is None
