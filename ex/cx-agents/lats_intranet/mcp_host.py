"""mcp_host.py, the plumbing between the notebook and the five intranet servers.

Plumbing only, nothing to learn from: MCP clients are asynchronous, and the agent loops in
the notebook are ordinary functions. `Host` runs the clients on an event loop in a
background thread and offers plain calls in front of them, so `host.call(...)` works from a
notebook cell, a checker or the language model's prompt alike.

The servers are built in the notebook (`make_site_server`), which is the part worth reading.
Here a server is any `mcp.server.mcpserver.MCPServer`; the host connects to it in-process,
which speaks the same protocol messages as a server started as a separate program.
"""
import asyncio
import contextlib
import json
import logging
import threading

from mcp import Client

# A tool refusing a bad page is an answer, not an incident: keep the servers' logs quiet.
logging.getLogger("mcp").setLevel(logging.CRITICAL)


class Host:
    """The AI application's side of MCP: one client per server, and a log of every request.

        host = Host({"people": people_server, ...})
        host.tools()                        tools/list on every server: {server: [tool, ...]}
        host.call("people", "read_page", {"page": 2})     tools/call, the structured result
        host.log                            the requests sent, oldest first
    """

    def __init__(self, servers):
        self.servers = dict(servers)
        self.log = []
        self._clients = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._ready = threading.Event()
        self._closing = None
        self._lifetime_task = asyncio.run_coroutine_threadsafe(self._lifetime(), self._loop)
        if not self._ready.wait(timeout=60):
            raise RuntimeError("the MCP clients did not connect within a minute")
        self._tools = {name: self._run(self._clients[name].list_tools()).tools
                       for name in self._clients}
        for name in self._clients:
            self.log.append({"server": name, "method": "tools/list"})

    async def _lifetime(self):
        """Open every client, wait for close(), close them: all in one task, as anyio requires."""
        self._closing = asyncio.Event()
        async with contextlib.AsyncExitStack() as stack:
            for name, server in self.servers.items():
                self._clients[name] = await stack.enter_async_context(Client(server))
                self.log.append({"server": name, "method": "initialize"})
            self._ready.set()
            await self._closing.wait()

    def _run(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=60)

    def tools(self):
        """What each server said it offers, as tools/list returned it."""
        return dict(self._tools)

    def call(self, server, tool, arguments):
        """tools/call on one server. The structured result, or {"error": text}."""
        if server not in self._clients:
            raise KeyError("no client for server %r: the host is not connected to it" % server)
        self.log.append({"server": server, "method": "tools/call",
                         "params": {"name": tool, "arguments": dict(arguments)}})
        result = self._run(self._clients[server].call_tool(tool, dict(arguments)))
        text = " ".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            return {"error": text}
        if result.structured_content is not None:
            return result.structured_content
        try:
            return json.loads(text)
        except ValueError:
            return {"text": text}

    def close(self):
        """Close every client and stop the background loop."""
        if self._closing is not None and not self._closing.is_set():
            self._loop.call_soon_threadsafe(self._closing.set)
            self._lifetime_task.result(timeout=60)
        self._loop.call_soon_threadsafe(self._loop.stop)


def tool_list_text(host):
    """The tools as the model reads them: one line per tool, from tools/list."""
    lines = ["Tools you can call (one MCP server per site):"]
    for server, tools in host.tools().items():
        for tool in tools:
            params = ", ".join("%s: %s" % (k, v.get("type", "any"))
                               for k, v in tool.input_schema.get("properties", {}).items())
            lines.append("  %s.%s(%s): %s" % (server, tool.name, params, tool.description))
    return "\n".join(lines)


def attach(host, lm, route):
    """Put the model behind `host`: its prompt lists the servers' tools, every page it reads
    comes back from tools/call, and each candidate action is shown as the call `route`
    turns it into. Actions no connected server can serve leave the menu."""
    import hf_lm
    import intranet

    cache = {}

    def fetch_page(site_key, index):
        if (site_key, index) not in cache:
            got = host.call(site_key, "read_page", {"page": index})
            if "error" in got:
                raise RuntimeError(got["error"])
            cache[(site_key, index)] = intranet.Page(
                got["title"], tuple(got["body"]),
                tuple(intranet.Link(l["label"], l["site"], l["page"]) for l in got["links"]))
        return cache[(site_key, index)]

    def render(state, action):
        call = route(state, action)
        if call is None or call[0] not in host.tools():
            return None
        server, tool, arguments = call
        args = ", ".join("%s=%r" % (k, v) for k, v in arguments.items())
        return "%s.%s(%s)" % (server, tool, args)

    hf_lm.TOOL_LIST = tool_list_text(host)
    hf_lm.FETCH_PAGE = fetch_page
    hf_lm.RENDER = render
    if hasattr(lm, "_cache"):
        lm._cache.clear()          # cached scores were computed from the old prompt


def detach(lm=None):
    """Back to the intranet read directly, as before Part 2."""
    import hf_lm
    hf_lm.TOOL_LIST = hf_lm.FETCH_PAGE = hf_lm.RENDER = None
    if lm is not None and hasattr(lm, "_cache"):
        lm._cache.clear()
