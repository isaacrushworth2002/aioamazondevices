# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Do Not Disturb push-event handling in AmazonEchoApi."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aioamazondevices.api import AmazonEchoApi

from .const import TEST_SERIAL_1


def _dnd_payload(serial: str, enabled: bool) -> dict[str, object]:
    """Build a DND push payload."""
    return {"dopplerId": {"deviceSerialNumber": serial}, "enabled": enabled}


@pytest.mark.anyio
async def test_dnd_event_triggers_initial_sync_once(
    api: AmazonEchoApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent push events perform the initial full sync only once."""

    async def sync_do_not_disturb_status() -> None:
        await asyncio.sleep(0)
        api._dnd_handler._dnd_states = {TEST_SERIAL_1: False}

    sync = AsyncMock(side_effect=sync_do_not_disturb_status)
    monkeypatch.setattr(api._dnd_handler, "sync_do_not_disturb_status", sync)

    await asyncio.gather(
        api._handle_dnd_event(_dnd_payload(TEST_SERIAL_1, True)),
        api._handle_dnd_event(_dnd_payload(TEST_SERIAL_1, True)),
    )

    sync.assert_awaited_once()


@pytest.mark.anyio
async def test_dnd_event_not_overwritten_by_in_flight_sync(
    api: AmazonEchoApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A push event arriving mid-sync survives the stale full-sync response."""
    api._dnd_initialized = True
    api._dnd_handler._dnd_states = {TEST_SERIAL_1: False}

    started = asyncio.Event()
    release = asyncio.Event()

    async def sync_do_not_disturb_status() -> None:
        started.set()
        await release.wait()
        # Snapshot taken before the push event: DND still disabled.
        api._dnd_handler._dnd_states = {TEST_SERIAL_1: False}

    monkeypatch.setattr(
        api._dnd_handler, "sync_do_not_disturb_status", sync_do_not_disturb_status
    )

    sync_task = asyncio.create_task(api.sync_dnd_state())
    await started.wait()

    # The push event lands while the full sync response is still in flight.
    event_task = asyncio.create_task(
        api._handle_dnd_event(_dnd_payload(TEST_SERIAL_1, True))
    )
    await asyncio.sleep(0)

    release.set()
    await asyncio.gather(sync_task, event_task)

    assert api._dnd_handler.dnd_states == {TEST_SERIAL_1: True}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"dopplerId": {"deviceSerialNumber": TEST_SERIAL_1}}, id="no-enabled"
        ),
        pytest.param(
            {"dopplerId": {"deviceSerialNumber": TEST_SERIAL_1}, "enabled": "true"},
            id="enabled-not-bool",
        ),
        pytest.param({"enabled": True}, id="no-serial"),
    ],
)
async def test_dnd_event_unusable_payload_is_ignored(
    api: AmazonEchoApi, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    """A payload without a boolean 'enabled' or a serial changes nothing."""
    sync = AsyncMock()
    monkeypatch.setattr(api._dnd_handler, "sync_do_not_disturb_status", sync)

    await api._handle_dnd_event(payload)

    sync.assert_not_awaited()
    assert api._dnd_handler.dnd_states == {}
