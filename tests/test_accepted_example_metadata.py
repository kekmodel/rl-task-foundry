from rl_task_foundry.tasks.models import AcceptedExample, SolverResult, TaskSpec, VerifyResult
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema, GroundTruth


def _sample_task() -> TaskSpec:
    return TaskSpec(
        task_id="task_1",
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
                )
            ]
        ),
        selected_path_id="orders.shipments",
        required_hops=2,
        tool_bundle_id="orders.shipments",
        sensitivity_policy="default",
    )


def _sample_truth() -> GroundTruth:
    return GroundTruth(
        task_id="task_1",
        verification_sql="SELECT 1",
        canonical_answer={"delivery_status": "in_transit"},
        answer_schema_version="v1",
    )


def test_accepted_example_includes_mean_correct_solver_turns_metadata():
    example = AcceptedExample(
        task=_sample_task(),
        ground_truth=_sample_truth(),
        solver_results=[
            SolverResult(
                task_id="task_1",
                solver_id="solver_a",
                provider="codex_oauth",
                model="gpt-5.4-mini",
                replica_index=0,
                transcript_ref="trace://transcript/a",
                tool_trace_ref="trace://tools/a",
                raw_output_text="{}",
                structured_output={"delivery_status": "in_transit"},
                status="completed",
                turn_count=5,
            ),
            SolverResult(
                task_id="task_1",
                solver_id="solver_b",
                provider="codex_oauth",
                model="gpt-5.4-mini",
                replica_index=1,
                transcript_ref="trace://transcript/b",
                tool_trace_ref="trace://tools/b",
                raw_output_text="{}",
                structured_output={"delivery_status": "in_transit"},
                status="completed",
                turn_count=2,
            ),
            SolverResult(
                task_id="task_1",
                solver_id="solver_c",
                provider="codex_oauth",
                model="gpt-5.4-mini",
                replica_index=2,
                transcript_ref="trace://transcript/c",
                tool_trace_ref="trace://tools/c",
                raw_output_text="{}",
                structured_output={"delivery_status": "lost"},
                status="completed",
                turn_count=9,
            ),
        ],
        verification_results=[
            VerifyResult(task_id="task_1", solver_id="solver_a", pass_exact=True),
            VerifyResult(task_id="task_1", solver_id="solver_b", pass_exact=True),
            VerifyResult(
                task_id="task_1",
                solver_id="solver_c",
                pass_exact=False,
                failure_reason="field_mismatch",
            ),
        ],
        pass_rate=2 / 3,
        calibration_band=(0.3, 0.8),
        export_payload={"task_id": "task_1"},
    )

    assert example.mean_correct_solver_turns_rounded == 4
    assert example.training_metadata == {"mean_correct_solver_turns_rounded": 4}

    payload = example.model_dump(mode="python")
    assert payload["mean_correct_solver_turns_rounded"] == 4
    assert payload["training_metadata"] == {"mean_correct_solver_turns_rounded": 4}


def test_accepted_example_omits_turn_metadata_when_no_correct_solver():
    example = AcceptedExample(
        task=_sample_task(),
        ground_truth=_sample_truth(),
        solver_results=[
            SolverResult(
                task_id="task_1",
                solver_id="solver_a",
                provider="codex_oauth",
                model="gpt-5.4-mini",
                replica_index=0,
                transcript_ref="trace://transcript/a",
                tool_trace_ref="trace://tools/a",
                raw_output_text="{}",
                structured_output={"delivery_status": "lost"},
                status="completed",
                turn_count=6,
            )
        ],
        verification_results=[
            VerifyResult(
                task_id="task_1",
                solver_id="solver_a",
                pass_exact=False,
                failure_reason="field_mismatch",
            )
        ],
        pass_rate=0.0,
        calibration_band=(0.3, 0.8),
        export_payload={"task_id": "task_1"},
    )

    assert example.mean_correct_solver_turns_rounded is None
    assert example.training_metadata == {}
