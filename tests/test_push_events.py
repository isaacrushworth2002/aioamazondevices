# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for dispatching HTTP2 push events in AmazonEchoApi."""

from unittest.mock import AsyncMock, call

import pytest

from aioamazondevices.api import AmazonEchoApi
from aioamazondevices.structures import AmazonPushMessage

PAYLOAD = {"key": "value"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("event_type", "handler", "expected_args"),
    [
        pytest.param(
            AmazonPushMessage.VolumeChange,
            "_handle_volume_change_event",
            (PAYLOAD,),
            id="volume",
        ),
        pytest.param(
            AmazonPushMessage.EqualizerStateChange,
            "_handle_eq_event_as_history_proxy",
            (),
            id="equalizer",
        ),
        pytest.param(
            AmazonPushMessage.AudioPlayerState,
            "_handle_audio_player_state_event",
            (),
            id="audio-player",
        ),
        pytest.param(
            AmazonPushMessage.ItemChange,
            "_handle_item_change_event",
            (PAYLOAD,),
            id="list-item",
        ),
        pytest.param(
            AmazonPushMessage.DoNotDisturbChange,
            "_handle_dnd_event",
            (PAYLOAD,),
            id="do-not-disturb",
        ),
        pytest.param(
            AmazonPushMessage.SmartHome,
            "_handle_smart_home_event",
            (PAYLOAD,),
            id="smart-home",
        ),
    ],
)
async def test_push_event_is_dispatched(
    api: AmazonEchoApi,
    monkeypatch: pytest.MonkeyPatch,
    event_type: AmazonPushMessage,
    handler: str,
    expected_args: tuple[dict[str, str], ...],
) -> None:
    """Each push event type is passed to its own handler."""
    handle = AsyncMock()
    monkeypatch.setattr(api, handler, handle)

    await api._http2_push_event_handler(event_type.value, PAYLOAD)

    assert handle.await_args_list == [call(*expected_args)]


@pytest.mark.anyio
async def test_unknown_push_event_is_ignored(
    api: AmazonEchoApi, debug_logs: pytest.LogCaptureFixture
) -> None:
    """Push events of an unknown type are logged and otherwise ignored."""
    await api._http2_push_event_handler("UNKNOWN_EVENT", {})

    assert "Unhandled push event type: UNKNOWN_EVENT" in debug_logs.text
