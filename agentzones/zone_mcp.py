"""The zone as an MCP server: plug your IDE assistant into the zone network.

Exposed at ``POST /mcp`` (MCP streamable HTTP, JSON responses) on the zone,
and over stdio via ``agentzone mcp`` (a thin bridge to that endpoint).

    # Claude Code / VS Code (HTTP)
    claude mcp add --transport http agentzones http://127.0.0.1:7700/mcp
    # Claude Desktop (stdio)  "agentzones": {"command": "agentzone", "args": ["mcp"]}

An assistant connected this way can list zones, ask agents in this or any
peer zone, call this zone's tools and peers' *exposed* tools, and search
verified fixes. It acts as the local user: same rules as the /api/* API
(loopback or ``security.api_token``). Peers still only see what they always
see: signed /p2p/* requests limited to public agents and exposed tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .mcp_lite import MiniMCPServer

if TYPE_CHECKING:
    from .zone import Zone


def _slim(res: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky internals from agent answers so they fit an assistant's context."""
    out = {k: v for k, v in res.items() if k not in ("trace", "steps")}
    if "trace" in res:
        out["trace_steps"] = [t.get("step") for t in res["trace"]]
    return out


def build_zone_mcp(zone: "Zone") -> MiniMCPServer:
    srv = MiniMCPServer(
        f"agentzone-{zone.name}",
        instructions=(f"Gateway to Agent Zone '{zone.name}' and its peer zones (self-hosted AI agents with local "
                      "models). Use list_zones first. ask_agent routes a question to an agent in this zone or a "
                      "peer zone; call_tool runs a tool here or a tool a peer zone exposes. For corporate proxy, "
                      "TLS or certificate errors, use diagnose_proxy."))

    def is_local(z: str) -> bool:
        return not z or z == zone.name

    @srv.tool()
    async def list_zones() -> list:
        """This zone and all reachable peer zones, with their agents and exposed tools."""
        me = zone.card()
        rows = [{"zone": zone.name, "local": True, "agents": [a.card() for a in zone.agents.values()],
                 "tags": me["tags"], "model": me["model"]}]
        for p in zone.peers.alive():
            rows.append({"zone": p.name, "local": False, "url": p.url, "agents": p.card.get("agents", []),
                         "exposed_tools": [t["name"] for t in p.card.get("tools", [])],
                         "tags": p.card.get("tags", []), "model": p.card.get("model")})
        return rows

    @srv.tool()
    async def ask_agent(agent: str, question: str, zone_name: str = "", context: dict | None = None) -> dict:
        """Ask an agent. zone_name empty = this zone; otherwise a peer zone from list_zones.
        context: optional JSON object passed to the agent."""
        if is_local(zone_name):
            return _slim(await zone.ask(agent, question, context))
        answers = await zone.ask_peers(question, context, agent=agent, zones=[zone_name])
        if not answers:
            raise ValueError(f"zone {zone_name!r} is unknown, offline, or has no agent {agent!r}")
        return _slim(answers[0])

    @srv.tool()
    async def ask_all_zones(question: str, agent: str = "", context: dict | None = None) -> list:
        """Ask the same question to one agent in every peer zone (default: the first public agent)."""
        return [_slim(a) for a in await zone.ask_peers(question, context, agent=agent or None)]

    @srv.tool()
    async def list_tools(zone_name: str = "") -> list:
        """Tools of this zone (all of them), or the tools a peer zone exposes."""
        if is_local(zone_name):
            return [{**t.spec(), "exposed_to_peers": t.exposed} for t in zone.tools.tools.values()]
        p = zone.peers.get(zone_name)
        if not p:
            raise ValueError(f"unknown or offline zone {zone_name!r}")
        return p.card.get("tools", [])

    @srv.tool()
    async def call_tool(tool: str, arguments: dict | None = None, zone_name: str = "") -> Any:
        """Call a tool by qualified name (e.g. proxy.check_https). zone_name empty = this zone;
        otherwise the tool must be exposed by that peer zone."""
        if is_local(zone_name):
            return await zone.tools.call(tool, arguments or {})
        return await zone.call_peer_tool(zone_name, tool, arguments or {})

    @srv.tool()
    async def search_fixes(problem: str = "", text: str = "") -> list:
        """Search verified fixes (signed by the zones that verified them) known to this zone."""
        return [{k: f[k] for k in ("id", "problem", "summary", "steps", "verified_by", "origin")}
                for f in zone.fixes.search(problem or None, text)]

    if "proxy-doctor" in zone.agents:
        @srv.tool()
        async def diagnose_proxy(problem_description: str, target: str = "", internal_url: str = "",
                                 simulate: str = "") -> dict:
            """Diagnose (and, if allowed, fix) a corporate proxy / TLS / certificate problem on THIS machine.
            problem_description: the error the user sees. target: URL that fails (default https://pypi.org).
            simulate: demo only - a failure class to simulate."""
            ctx = {k: v for k, v in {"target": target, "internal_url": internal_url, "simulate": simulate,
                                     "error_text": problem_description}.items() if v}
            return _slim(await zone.ask("proxy-doctor", problem_description, ctx))

    return srv
