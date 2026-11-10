# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for utility helpers."""

from collections.abc import Callable
from typing import Any

import pytest

from aioamazondevices.const.http import ARRAY_WRAPPER
from aioamazondevices.exceptions import CannotRetrieveData
from aioamazondevices.structures import AmazonDevice
from aioamazondevices.utils import (
    _MAX_JSON_PARSE_DEPTH,
    check_and_log_graphql_errors,
    get_innermost_frame,
    http2_extract_json_from_part,
    http2_parse_boundary_delimiter,
    obfuscate_email,
    parse_device_details,
    replace_routine_placeholders,
    scrub_fields,
    string_recursive_parse,
    validate_graphql_endpoint_shape,
)

from .const import TEST_DEVICE_TYPE, TEST_SERIAL_1

TEST_ENDPOINTS = [{"endpointId": "endpoint-1"}]


def _part(content_type: str, body: bytes) -> bytes:
    """Build one part of a multipart AVS response."""
    return f"Content-Type: {content_type}\r\n\r\n".encode() + body


@pytest.mark.parametrize(
    ("raw_model", "expected_model", "expected_hardware_version"),
    [
        pytest.param(
            "Echo Dot (5th gen) with Clock",
            "Echo Dot with Clock",
            "5th Gen",
            id="generation-mid-name",
        ),
        pytest.param(
            "Echo Dot (4th generation)", "Echo Dot", "4th Gen", id="generation-long"
        ),
        pytest.param("Echo Dot (3rd Gen)", "Echo Dot", "3rd Gen", id="generation"),
        pytest.param(
            "Fire TV Stick 4K (2nd Gen)",
            "Fire TV Stick 4K",
            "2nd Gen",
            id="generation-after-digits",
        ),
        pytest.param(
            "Echo Dot (5th gen)", "Echo Dot", "5th Gen", id="generation-lowercase"
        ),
        pytest.param("2021 Samsung UHD TV", "Samsung UHD TV", "2021", id="year-prefix"),
        pytest.param("LG OLED TV 2023", "LG OLED TV", "2023", id="year-suffix"),
        pytest.param(
            "Panasonic Viera (2019)", "Panasonic Viera", "2019", id="year-bracketed"
        ),
        pytest.param(None, None, None, id="none"),
    ],
)
def test_parse_device_details(
    raw_model: str | None,
    expected_model: str | None,
    expected_hardware_version: str | None,
) -> None:
    """Hardware generations and years are split from the model name."""
    model, hardware_version = parse_device_details(raw_model)

    assert model == expected_model
    assert hardware_version == expected_hardware_version


@pytest.mark.parametrize(
    "response",
    [
        pytest.param({"data": {}}, id="data"),
        pytest.param({"errors": []}, id="empty-errors"),
        pytest.param({ARRAY_WRAPPER: [{"data": {}}]}, id="array-wrapped"),
    ],
)
def test_check_and_log_graphql_errors_no_errors(
    response: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Responses without errors are not flagged."""
    assert not check_and_log_graphql_errors("TEST_QUERY", response)
    assert not caplog.records


@pytest.mark.parametrize(
    ("response", "messages"),
    [
        pytest.param(
            {"errors": [{"message": "Failed", "path": ["listEndpoints"]}, {}]},
            [
                "Error running query: TEST_QUERY - Failed for path ['listEndpoints']",
                "Error running query: TEST_QUERY - Unknown error for path Unknown path",
            ],
            id="every-error",
        ),
        pytest.param(
            {ARRAY_WRAPPER: [{"errors": [{"message": "Boom", "path": "x"}]}]},
            ["Error running query: TEST_QUERY - Boom for path x"],
            id="array-wrapped",
        ),
    ],
)
def test_check_and_log_graphql_errors_logs_errors(
    response: dict[str, Any], messages: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Every error is logged with the query name, also inside the array wrapper."""
    assert check_and_log_graphql_errors("TEST_QUERY", response)
    assert [record.getMessage() for record in caplog.records] == messages


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param(
            {"data": {"listEndpoints": {"endpoints": TEST_ENDPOINTS}}},
            TEST_ENDPOINTS,
            id="bare",
        ),
        pytest.param(
            {
                ARRAY_WRAPPER: [
                    {"data": {"listEndpoints": {"endpoints": TEST_ENDPOINTS}}}
                ]
            },
            TEST_ENDPOINTS,
            id="array-wrapped",
        ),
        pytest.param(
            {"data": {"listEndpoints": {"endpoints": []}}}, [], id="empty-endpoints"
        ),
    ],
)
def test_validate_graphql_endpoint_shape_returns_endpoints(
    response: dict[str, Any],
    expected: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Endpoints are returned as is from bare and array wrapped responses."""
    assert validate_graphql_endpoint_shape("TEST_QUERY", response) == expected
    assert not caplog.records


@pytest.mark.parametrize(
    "response",
    [
        pytest.param({}, id="empty"),
        pytest.param({"data": None}, id="null-data"),
        pytest.param({"data": {}}, id="no-list-endpoints"),
        pytest.param({"data": {"listEndpoints": None}}, id="null-list-endpoints"),
        pytest.param({"data": {"listEndpoints": {}}}, id="no-endpoints"),
        pytest.param(
            {"data": {"listEndpoints": {"endpoints": None}}}, id="null-endpoints"
        ),
        pytest.param(
            {"data": {"listEndpoints": {"endpoints": {"endpointId": "x"}}}},
            id="endpoints-not-a-list",
        ),
        pytest.param({ARRAY_WRAPPER: []}, id="empty-array"),
        pytest.param({ARRAY_WRAPPER: ["not a dict"]}, id="array-item-not-a-dict"),
        pytest.param({ARRAY_WRAPPER: [{"data": {}}]}, id="array-no-list-endpoints"),
    ],
)
def test_validate_graphql_endpoint_shape_malformed(
    response: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Malformed responses are logged with the query name."""
    assert validate_graphql_endpoint_shape("TEST_QUERY", response) is None
    assert "Malformed endpoint data received in TEST_QUERY" in caplog.text


def test_http2_extract_json_from_part() -> None:
    """A JSON part is parsed, including JSON nested inside its strings."""
    part = _part("application/json", b'{"directive": "{\\"name\\": \\"Speak\\"}"}')

    assert http2_extract_json_from_part(part) == {"directive": {"name": "Speak"}}


@pytest.mark.parametrize(
    "part",
    [
        pytest.param(_part("text/plain", b'{"a": 1}'), id="not-json-content"),
        pytest.param(_part("application/json", b"{not json"), id="invalid-json"),
        pytest.param(_part("application/json", b"[1, 2]"), id="not-an-object"),
    ],
)
def test_http2_extract_json_from_part_rejects(
    part: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    """Parts that are not a JSON object are logged and skipped."""
    assert http2_extract_json_from_part(part) is None
    assert "Failed to parse multipart section" in caplog.text


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        pytest.param("first.last@example.com", "f****.l***@e******.com", id="dotted"),
        pytest.param("a@b.com", "*@*.com", id="single-characters"),
        pytest.param("not-an-email", "[invalid email]", id="no-at"),
        pytest.param("user@localhost", "[invalid email]", id="no-domain-suffix"),
    ],
)
def test_obfuscate_email(email: str, expected: str) -> None:
    """All but the first character of each part of an address is hidden."""
    assert obfuscate_email(email) == expected


def test_scrub_fields() -> None:
    """Sensitive keys are redacted at any depth, and emails are obfuscated."""
    data = {
        "email": "user@example.com",
        "password": "secret",
        "nested": [{"token": "t"}, ({"name": "n"},), {"kept"}],
        "kept": 1,
    }

    assert scrub_fields(data) == {
        "email": "u***@e******.com",
        "password": "[REDACTED]",
        "nested": [{"token": "[REDACTED]"}, ({"name": "[REDACTED]"},), {"kept"}],
        "kept": 1,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            {"a": '{"b": "[1, 2]"}'}, {"a": {"b": [1, 2]}}, id="nested-strings"
        ),
        pytest.param(["{not json", "text"], ["{not json", "text"], id="invalid-json"),
        pytest.param('{"a": 1}', {"a": 1}, id="top-level-string"),
    ],
)
def test_string_recursive_parse(
    value: dict[str, Any] | list[Any] | str, expected: object
) -> None:
    """Strings holding JSON are parsed, other strings are kept."""
    assert string_recursive_parse(value) == expected


def test_string_recursive_parse_stops_at_max_depth() -> None:
    """JSON nested deeper than the limit is left as a string."""
    deepest = '{"a": 1}'
    value: dict[str, Any] | str = deepest
    for _ in range(_MAX_JSON_PARSE_DEPTH):
        value = {"a": value}

    parsed = string_recursive_parse(value)
    for _ in range(_MAX_JSON_PARSE_DEPTH):
        assert isinstance(parsed, dict)
        parsed = parsed["a"]

    assert parsed == deepest


def test_replace_routine_placeholders(make_device: Callable[..., AmazonDevice]) -> None:
    """Device placeholders are replaced in both forms, other values kept."""
    trigger = "<$Trigger.Alexa.Trigger.Alarms.NotificationStopped.$.payload.object"
    routine = {
        "nodes": [
            {
                "deviceSerialNumber": "ALEXA_CURRENT_DSN",
                "deviceType": "ALEXA_CURRENT_DEVICE_TYPE",
            },
            {
                "deviceSerialNumber": f"{trigger}.device.deviceSerialNumber$>",
                "deviceType": f"{trigger}.device.productId$>",
            },
        ],
        "volume": 50,
        "text": "Good morning",
    }

    assert replace_routine_placeholders(routine, make_device(TEST_SERIAL_1)) == {
        "nodes": [
            {"deviceSerialNumber": TEST_SERIAL_1, "deviceType": TEST_DEVICE_TYPE},
            {"deviceSerialNumber": TEST_SERIAL_1, "deviceType": TEST_DEVICE_TYPE},
        ],
        "volume": 50,
        "text": "Good morning",
    }


def test_get_innermost_frame_inside_library() -> None:
    """The innermost frame inside the library is named with its line."""
    with pytest.raises(CannotRetrieveData) as exc_info:
        http2_parse_boundary_delimiter("application/json")

    frame = get_innermost_frame(exc_info.value)
    assert frame.startswith("http2_parse_boundary_delimiter:")


def test_get_innermost_frame_outside_library() -> None:
    """An error raised outside the library has no library frame."""

    def fail() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom") as exc_info:
        fail()

    assert get_innermost_frame(exc_info.value) == "unknown"
