from rl_task_foundry.tasks.models import TaskSpec
from rl_task_foundry.tasks.validator import TaskValidator
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema


def test_task_validator_requires_answer_schema_and_path():
    task = TaskSpec(
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

    issues = TaskValidator().validate(task)
    assert issues == []
