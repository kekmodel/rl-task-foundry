from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.infra.db import smoke_test_connection


@pytest.mark.asyncio
async def test_smoke_test_connection_hits_pagila_container():
    config = load_config(Path("rl_task_foundry.yaml"))
    info = await smoke_test_connection(config.database)
    assert info["database_name"] == "pagila"
    assert info["user_name"] == config.database.readonly_role
    assert info["read_only"] == "on"
