# Copyright 2024 Simone Chemelli and contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the HTTP wrapper."""

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from http import HTTPMethod, HTTPStatus
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import ClientConnectionError, ContentTypeError, web
from aiohttp.test_utils import TestServer
from yarl import URL

from aioamazondevices import http_wrapper as http_wrapper_module
from aioamazondevices.api import AmazonEchoApi
from aioamazondevices.const.http import (
    ARRAY_WRAPPER,
    CSRF_COOKIE,
    HTTP_ERROR_199,
    HTTP_ERROR_299,
    REFRESH_ACCESS_TOKEN,
    REFRESH_AUTH_COOKIES,
    URI_CAPABILITIES,
    URI_SIGNIN,
)
from aioamazondevices.exceptions import (
    CannotAuthenticate,
    CannotConnect,
    CannotRetrieveData,
)

from .const import TEST_CSRF, TEST_REFRESH_TOKEN


def _app(
    path: str, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> web.Application:
    """Build an app serving one GET route."""
    app = web.Application()
    app.router.add_get(path, handler)
    return app


def _status(status: int) -> Callable[[web.Request], Awaitable[web.Response]]:
    """Return a handler answering with an empty response of the given status."""

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=status)

    return handler


@pytest.mark.anyio
async def test_csrf_cookie_sent_on_next_request(api: AmazonEchoApi) -> None:
    """A CSRF cookie set by a response is sent as a header on later requests."""
    received_csrf: list[str | None] = []

    async def handler(request: web.Request) -> web.Response:
        received_csrf.append(request.headers.get(CSRF_COOKIE))
        response = web.Response(text="<html></html>", content_type="text/html")
        response.set_cookie(CSRF_COOKIE, TEST_CSRF)
        return response

    app = web.Application()
    app.router.add_get("/", handler)
    async with TestServer(app) as server:
        url = server.make_url("/")
        await api._http_wrapper.session_request(HTTPMethod.GET, url)
        await api._http_wrapper.session_request(HTTPMethod.GET, url)

    assert received_csrf == [None, TEST_CSRF]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("fail_fast", "expected_sleeps", "expected_requests"),
    [
        pytest.param(False, [2, 5], 3, id="retries-with-backoff"),
        pytest.param(True, [], 1, id="fail-fast-does-not-retry"),
    ],
)
async def test_retry_behaviour(
    api: AmazonEchoApi,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_fast: bool,
    expected_sleeps: list[float],
    expected_requests: int,
) -> None:
    """Retryable statuses back off and retry, unless fail_fast is set."""
    sleeps: list[float] = []
    requests: list[web.Request] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(http_wrapper_module.asyncio, "sleep", sleep)

    async def handler(request: web.Request) -> web.Response:
        requests.append(request)
        return web.Response(status=HTTPStatus.SERVICE_UNAVAILABLE)

    app = web.Application()
    app.router.add_get("/", handler)
    async with TestServer(app) as server:
        with pytest.raises(CannotRetrieveData):
            await api._http_wrapper.session_request(
                HTTPMethod.GET, server.make_url("/"), fail_fast=fail_fast
            )

    # Aiohttp internals also yield with sleep(0), so only the delays are compared.
    assert [delay for delay in sleeps if delay] == expected_sleeps
    assert len(requests) == expected_requests


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        pytest.param(HTTP_ERROR_199, "Miscellaneous Warning", id="199"),
        pytest.param(HTTP_ERROR_299, "Miscellaneous Persistent Warning", id="299"),
        pytest.param(HTTPStatus.NOT_FOUND, "Not Found", id="standard"),
    ],
)
async def test_http_phrase_error(
    api: AmazonEchoApi, status: int, expected: str
) -> None:
    """Amazon's non-standard warning codes have their own phrases."""
    assert await api._http_wrapper.http_phrase_error(status) == expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status",
    [
        pytest.param(HTTPStatus.UNAUTHORIZED, id="unauthorized"),
        pytest.param(HTTPStatus.FORBIDDEN, id="forbidden"),
        pytest.param(HTTPStatus.PROXY_AUTHENTICATION_REQUIRED, id="proxy-auth"),
    ],
)
async def test_auth_failure_raises(api: AmazonEchoApi, status: HTTPStatus) -> None:
    """Authentication failures raise CannotAuthenticate with the status phrase."""
    async with TestServer(_app("/", _status(status))) as server:
        with pytest.raises(CannotAuthenticate, match=status.phrase):
            await api._http_wrapper.session_request(
                HTTPMethod.GET, server.make_url("/")
            )


@pytest.mark.anyio
async def test_signin_not_found_after_redirect_is_ignored(api: AmazonEchoApi) -> None:
    """The sign-in page answers 404 after a redirect, which is not an error."""

    async def signin(_request: web.Request) -> web.Response:
        raise web.HTTPFound("/missing")

    app = _app(f"/{URI_SIGNIN}", signin)
    app.router.add_get("/missing", _status(HTTPStatus.NOT_FOUND))
    async with TestServer(app) as server:
        _, resp = await api._http_wrapper.session_request(
            HTTPMethod.GET, server.make_url(f"/{URI_SIGNIN}")
        )

    assert resp.status == HTTPStatus.NOT_FOUND


@pytest.mark.anyio
async def test_capabilities_no_content_is_ignored(api: AmazonEchoApi) -> None:
    """The capabilities endpoint answers 204, which is not an error."""
    path = f"/{URI_CAPABILITIES}"
    async with TestServer(_app(path, _status(HTTPStatus.NO_CONTENT))) as server:
        _, resp = await api._http_wrapper.session_request(
            HTTPMethod.GET, server.make_url(path)
        )

    assert resp.status == HTTPStatus.NO_CONTENT


@pytest.mark.anyio
async def test_not_found_elsewhere_raises(api: AmazonEchoApi) -> None:
    """A 404 from any other endpoint is an error."""
    async with TestServer(_app("/", _status(HTTPStatus.NOT_FOUND))) as server:
        with pytest.raises(CannotRetrieveData, match="Request failed: Not Found"):
            await api._http_wrapper.session_request(
                HTTPMethod.GET, server.make_url("/")
            )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(TimeoutError(), id="timeout"),
        pytest.param(ClientConnectionError("refused"), id="connection-error"),
    ],
)
async def test_connection_error_raises_cannot_connect(
    api: AmazonEchoApi, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Timeouts and connection errors raise CannotConnect without retrying."""
    request = AsyncMock(side_effect=error)
    monkeypatch.setattr(api._http_wrapper._session, "request", request)

    with pytest.raises(CannotConnect, match="Connection error during GET"):
        await api._http_wrapper.session_request(
            HTTPMethod.GET, URL("https://www.amazon.com/")
        )

    request.assert_awaited_once()


@pytest.mark.anyio
async def test_response_is_saved_with_callback(
    api: AmazonEchoApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a save callback configured, every response body is passed to it."""
    save = AsyncMock()
    save_data = api._http_wrapper._save_data
    monkeypatch.setattr(save_data, "callback", save)

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="<html></html>", content_type="text/html")

    async with TestServer(_app("/", page)) as server:
        url = server.make_url("/")
        await api._http_wrapper.session_request(HTTPMethod.GET, url)

    save.assert_awaited_once_with(
        save_data.path, "<html></html>", url.human_repr(), "text/html; charset=utf-8"
    )


@pytest.mark.anyio
async def test_logged_in_requests_use_website_cookies(api: AmazonEchoApi) -> None:
    """Once logged in, the stored website cookies are set for the account's domain.

    Outside amazon.com, they are also set for amazon.com, for the global endpoints.
    """
    state = api._session_state_data
    state.country_specific_data("https://www.amazon.co.uk")
    state.login_stored_data = {
        "website_cookies": {"ubid-acbuk": "UBID"},
        "store_authentication_cookie": {"cookie": "SESSION"},
    }

    async with TestServer(_app("/", _status(HTTPStatus.OK))) as server:
        await api._http_wrapper.session_request(HTTPMethod.GET, server.make_url("/"))

    expected = {"ubid-acbuk": "UBID", "session-token": "SESSION", "lc-acbit": "en-GB"}
    jar = api._http_wrapper._session.cookie_jar
    for site in ("https://www.amazon.co.uk", "https://www.amazon.com"):
        cookies = jar.filter_cookies(URL(site))
        assert {name: cookies[name].value for name in expected} == expected


@pytest.mark.anyio
async def test_clear_cookies(api: AmazonEchoApi) -> None:
    """Clearing cookies empties the jar and forgets the CSRF cookie."""
    wrapper = api._http_wrapper
    await wrapper.set_cookies({"name": "value"}, URL("https://www.amazon.com"))
    wrapper._csrf_cookie = TEST_CSRF

    await wrapper.clear_cookies()

    assert len(wrapper._session.cookie_jar) == 0
    assert wrapper._csrf_cookie is None


@pytest.mark.anyio
async def test_refresh_data_without_login_data(
    api: AmazonEchoApi, mock_json_response: Callable[..., AsyncMock]
) -> None:
    """Nothing can be refreshed before logging in."""
    session_request = mock_json_response({})

    assert await api._http_wrapper.refresh_data(REFRESH_ACCESS_TOKEN) == (False, {})
    session_request.assert_not_awaited()


@pytest.mark.anyio
async def test_refresh_data_request(
    api: AmazonEchoApi, mock_json_response: Callable[..., AsyncMock]
) -> None:
    """The refresh token is exchanged for the requested token type."""
    api._session_state_data.login_stored_data = {"refresh_token": TEST_REFRESH_TOKEN}
    session_request = mock_json_response({})

    await api._http_wrapper.refresh_data(REFRESH_AUTH_COOKIES)

    (request,) = session_request.await_args_list
    assert request.kwargs["method"] == HTTPMethod.POST
    assert request.kwargs["url"] == URL("https://api.amazon.com/auth/token")
    assert request.kwargs["json_data"] is False
    data = request.kwargs["input_data"]
    assert data["source_token"] == TEST_REFRESH_TOKEN
    assert data["requested_token_type"] == REFRESH_AUTH_COOKIES
    assert data["domain"] == "www.amazon.com"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("data_type", "response", "expected"),
    [
        pytest.param(
            REFRESH_ACCESS_TOKEN,
            (HTTPStatus.OK, {REFRESH_ACCESS_TOKEN: "NEW"}),
            ((True, {REFRESH_ACCESS_TOKEN: "NEW"}), "NEW"),
            id="access-token",
        ),
        pytest.param(
            REFRESH_ACCESS_TOKEN,
            (HTTPStatus.OK, {}),
            ((False, {}), "OLD"),
            id="access-token-missing",
        ),
        pytest.param(
            REFRESH_AUTH_COOKIES,
            (HTTPStatus.OK, {"response": {}}),
            ((True, {"response": {}}), "OLD"),
            id="auth-cookies",
        ),
        pytest.param(
            REFRESH_ACCESS_TOKEN,
            (HTTPStatus.BAD_REQUEST, {REFRESH_ACCESS_TOKEN: "NEW"}),
            ((False, {}), "OLD"),
            id="rejected",
        ),
    ],
)
async def test_refresh_data_result(
    api: AmazonEchoApi,
    mock_json_response: Callable[..., AsyncMock],
    data_type: str,
    response: tuple[HTTPStatus, dict[str, Any]],
    expected: tuple[tuple[bool, dict[str, Any]], str],
) -> None:
    """A new access token is stored, other refreshes are only returned."""
    status, json = response
    expected_result, expected_token = expected
    api._session_state_data.login_stored_data = {
        "refresh_token": TEST_REFRESH_TOKEN,
        REFRESH_ACCESS_TOKEN: "OLD",
    }
    mock_json_response(json, status=status)

    assert await api._http_wrapper.refresh_data(data_type) == expected_result
    stored = api._session_state_data.login_stored_data
    assert stored[REFRESH_ACCESS_TOKEN] == expected_token


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("json", "expected"),
    [
        pytest.param({"key": "value"}, {"key": "value"}, id="object"),
        pytest.param([1, 2], {ARRAY_WRAPPER: [1, 2]}, id="array"),
        pytest.param(None, {}, id="empty"),
    ],
)
async def test_response_to_json(
    api: AmazonEchoApi, json: object, expected: dict[str, Any]
) -> None:
    """Arrays are wrapped in an object, and empty responses become one."""
    raw_resp = MagicMock(json=AsyncMock(return_value=json))

    assert await api._http_wrapper.response_to_json(raw_resp, "") == expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("url", "error", "expected"),
    [
        pytest.param(
            f"https://www.amazon.com/{URI_SIGNIN}",
            ContentTypeError(MagicMock(), ()),
            pytest.raises(CannotAuthenticate, match="check credentials"),
            id="redirected-to-signin",
        ),
        pytest.param(
            "https://www.amazon.com/api",
            ContentTypeError(MagicMock(), ()),
            pytest.raises(ValueError, match="not in JSON format"),
            id="not-json",
        ),
        pytest.param(
            "https://www.amazon.com/api",
            orjson.JSONDecodeError("bad", "{", 0),
            pytest.raises(ValueError, match="corrupted JSON format"),
            id="corrupted-json",
        ),
    ],
)
async def test_response_to_json_errors(
    api: AmazonEchoApi,
    url: str,
    error: Exception,
    expected: AbstractContextManager[object],
) -> None:
    """Responses that are not JSON raise, as an auth failure on the sign-in page."""
    raw_resp = MagicMock(url=URL(url), json=AsyncMock(side_effect=error))

    with expected:
        await api._http_wrapper.response_to_json(raw_resp, "")
