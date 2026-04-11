# Review Pack

이 파일은 사람 눈으로 task 품질을 검토하기 위한 snapshot이다.

## 1. task::aggregate_verification::d201359c8c1e

- family: `aggregate_verification`
- outcome: `answer`
- tier/tool level: `A` / `L1`
- path: `payment.rental.customer`
- hops: `2`
- question strategy: `model_generated`

### Question

제 결제 내역이 총 몇 건인지 알려주세요.

### Submit Result Format

```json
{
  "version": "v1",
  "fields": [
    {
      "name": "payment_count",
      "type": "int",
      "nullable": false,
      "ordered": false,
      "canonicalizer": "int_cast",
      "description": "Count of 결제 items sharing the same 고객 context.",
      "visibility": "user_visible",
      "source_columns": [
        "meta:count"
      ]
    }
  ],
  "primary_output_format": "json_object"
}
```

### Tool Set

- `count_peer_payment_for_payment_by_customer_via_rental` (count, core): L1 count tool. Count how many related payment results share the same customer context.

### Review Notes

- seed question: 같은 고객에 연결된 결제가 몇 건인지 알려주세요.
- presentation strategy: canonical_rule_based
- difficulty features: `{"answer_shape": "count", "cardinality_estimate": 599.0, "condition_complexity": 1, "fanout_ambiguity": 1, "fanout_max": 26.784641068447414, "fanout_product": 26.792988313856426, "has_nullable_hop": false, "has_unique_join": false, "required_hops": 2, "shortcut_count": 1}`

<details>
<summary>Answer Key</summary>

```json
{
  "task_id": "task::aggregate_verification::d201359c8c1e",
  "verification_sql": "SELECT COUNT(DISTINCT r3.\"payment_id\")::bigint AS count FROM \"public\".\"payment\" AS t0 JOIN \"public\".\"rental\" AS t1 ON t0.\"rental_id\" = t1.\"rental_id\" JOIN \"public\".\"customer\" AS t2 ON t1.\"customer_id\" = t2.\"customer_id\" JOIN \"public\".\"payment\" AS r3 ON r3.\"customer_id\" = t2.\"customer_id\" WHERE t0.\"payment_id\" = :anchor_payment_id AND r3.\"payment_id\" IS NOT NULL",
  "sql_params": {
    "anchor_payment_id": 5329
  },
  "canonical_answer": {
    "payment_count": 40
  },
  "row_context": [
    {
      "count": 40
    }
  ],
  "answer_schema_version": "v1",
  "provenance_path": [
    "payment",
    "rental",
    "customer"
  ]
}
```

</details>

## 2. task::aggregate_verification::661e694c9497

- family: `aggregate_verification`
- outcome: `answer`
- tier/tool level: `A` / `L1`
- path: `customer.address.city`
- hops: `2`
- question strategy: `model_generated`

### Question

제가 등록한 주소들 중 같은 도시에 있는 주소가 몇 개인지 알려주세요.

### Submit Result Format

```json
{
  "version": "v1",
  "fields": [
    {
      "name": "address_count",
      "type": "int",
      "nullable": false,
      "ordered": false,
      "canonicalizer": "int_cast",
      "description": "Count of 주소 items sharing the same 도시 context.",
      "visibility": "user_visible",
      "source_columns": [
        "meta:count"
      ]
    }
  ],
  "primary_output_format": "json_object"
}
```

### Tool Set

- `count_address_for_customer_by_city_via_address` (count, core): L1 count tool. Count how many related address results share the same city context.

### Review Notes

- seed question: 같은 도시에 연결된 주소가 몇 개인지 알려주세요.
- presentation strategy: canonical_rule_based
- difficulty features: `{"answer_shape": "count", "cardinality_estimate": 600.0, "condition_complexity": 1, "fanout_ambiguity": 0, "fanout_max": 1.005, "fanout_product": 0.9983333333333333, "has_nullable_hop": false, "has_unique_join": false, "required_hops": 2, "shortcut_count": 0}`

<details>
<summary>Answer Key</summary>

```json
{
  "task_id": "task::aggregate_verification::661e694c9497",
  "verification_sql": "SELECT COUNT(DISTINCT r3.\"address_id\")::bigint AS count FROM \"public\".\"customer\" AS t0 JOIN \"public\".\"address\" AS t1 ON t0.\"address_id\" = t1.\"address_id\" JOIN \"public\".\"city\" AS t2 ON t1.\"city_id\" = t2.\"city_id\" JOIN \"public\".\"address\" AS r3 ON r3.\"city_id\" = t2.\"city_id\" WHERE t0.\"customer_id\" = :anchor_customer_id AND r3.\"address_id\" IS NOT NULL",
  "sql_params": {
    "anchor_customer_id": 252
  },
  "canonical_answer": {
    "address_count": 2
  },
  "row_context": [
    {
      "count": 2
    }
  ],
  "answer_schema_version": "v1",
  "provenance_path": [
    "customer",
    "address",
    "city"
  ]
}
```

</details>
