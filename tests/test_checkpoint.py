"""Tests for checkpoint with WAL-based durability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlvr_synth.checkpoint import RunCheckpoint


@pytest.mark.asyncio
async def test_mark_and_check_processed(tmp_path: Path) -> None:
    cp = RunCheckpoint(output_dir=tmp_path)
    assert not cp.is_processed(42)

    await cp.mark_processed(42)
    assert cp.is_processed(42)


@pytest.mark.asyncio
async def test_persistence_across_instances(tmp_path: Path) -> None:
    cp1 = RunCheckpoint(output_dir=tmp_path)
    await cp1.mark_processed(1)
    await cp1.mark_processed(2)

    # New instance loads from disk
    cp2 = RunCheckpoint(output_dir=tmp_path)
    assert cp2.is_processed(1)
    assert cp2.is_processed(2)
    assert not cp2.is_processed(3)


@pytest.mark.asyncio
async def test_manifest(tmp_path: Path) -> None:
    cp = RunCheckpoint(output_dir=tmp_path, run_id="test_run", rng_seed=42)
    await cp.mark_processed(10)
    manifest = cp.get_manifest()

    assert manifest["run_id"] == "test_run"
    assert manifest["rng_seed"] == 42
    assert 10 in manifest["processed_pks"]
