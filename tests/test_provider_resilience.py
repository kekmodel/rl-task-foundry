from rl_task_foundry.pipeline.provider_resilience import ProviderCircuitBreaker


def test_provider_circuit_breaker_trips_and_recovers():
    breaker = ProviderCircuitBreaker(
        provider_name="codex_oauth",
        window_s=60,
        threshold=0.3,
        probe_interval_s=30,
        minimum_request_count=2,
    )

    breaker.record_failure(now=0.0)
    assert breaker.is_available(now=0.0)

    breaker.record_failure(now=1.0)
    assert not breaker.is_available(now=2.0)

    snapshot = breaker.snapshot(now=2.0)
    assert snapshot.failures == 2
    assert snapshot.total_requests == 2
    assert snapshot.error_rate == 1.0
    assert snapshot.cooldown_remaining_s > 0

    assert breaker.is_available(now=31.0)
    breaker.record_success(now=31.0)
    assert breaker.is_available(now=31.0)
