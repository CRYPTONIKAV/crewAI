"""Pagination of native MCP tool discovery (issue #8098).

Behavior: _list_tools_impl follows each returned nextCursor until the
response omits it, returning every advertised tool.
"""

from types import SimpleNamespace

import pytest

from crewai.mcp.client import MCPClient
from crewai.mcp.transports.base import BaseTransport, TransportType


class _FakeTransport(BaseTransport):
    transport_type = TransportType.STDIO

    def __init__(self):
        super().__init__(command="fake", args=[])

    async def connect(self):
        return None

    async def disconnect(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


def _tool(name):
    return SimpleNamespace(name=name, description=f"{name} tool", inputSchema={})


class _PagedSession:
    """Fake SDK session serving two pages then omitting the cursor."""

    def __init__(self):
        self.cursors_seen = []

    async def list_tools(self, params=None):
        cursor = getattr(params, "cursor", None)
        self.cursors_seen.append(cursor)
        if cursor is None:
            return SimpleNamespace(
                tools=[_tool("search-tickets")], nextCursor="page-2"
            )
        assert cursor == "page-2"
        return SimpleNamespace(tools=[_tool("get-ticket")], nextCursor=None)


def _client_with_session(session):
    client = MCPClient(_FakeTransport())
    client._session = session
    return client


@pytest.mark.asyncio
async def test_list_tools_impl_follows_next_cursor():
    session = _PagedSession()
    client = _client_with_session(session)
    tools = await client._list_tools_impl()
    assert [t["name"] for t in tools] == ["search_tickets", "get_ticket"]
    assert session.cursors_seen == [None, "page-2"]


@pytest.mark.asyncio
async def test_list_tools_impl_single_page_without_cursor():
    class _SinglePage(_PagedSession):
        async def list_tools(self, params=None):
            self.cursors_seen.append(getattr(params, "cursor", None))
            return SimpleNamespace(tools=[_tool("only")], nextCursor=None)

    session = _SinglePage()
    client = _client_with_session(session)
    tools = await client._list_tools_impl()
    assert [t["name"] for t in tools] == ["only"]
    assert session.cursors_seen == [None]
