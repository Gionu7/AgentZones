import sys

from agentzones.mcp_lite import StdioMCPClient


async def test_proxy_doctor_mcp_server_over_stdio():
    c = StdioMCPClient("proxy", sys.executable, ["-m", "proxy_doctor.mcp_server"])
    await c.start()
    try:
        names = {t.name for t in c.tools}
        assert {"diagnose", "apply_fix", "propose_fix", "check_https", "inspect_tls"} <= names
        diag = next(t for t in c.tools if t.name == "diagnose")
        assert diag.input_schema["properties"]["fresh"]["type"] == "boolean"

        import json
        text, err = await c.call_tool("diagnose", {"simulate": "untrusted_corp_ca"})
        obs = json.loads(text)
        assert not err and obs["simulated"] and obs["external"]["ok"] is False

        # wrong fix does not cure the simulated machine
        await c.call_tool("apply_fix", {"problem": "clock_skew", "dry_run": False})
        obs = json.loads((await c.call_tool("diagnose", {"simulate": "untrusted_corp_ca", "fresh": False}))[0])
        assert obs["external"]["ok"] is False

        await c.call_tool("apply_fix", {"problem": "untrusted_corp_ca", "dry_run": False})
        obs = json.loads((await c.call_tool("diagnose", {"simulate": "untrusted_corp_ca", "fresh": False}))[0])
        assert obs["external"]["ok"] is True

        text, err = await c.call_tool("propose_fix", {"problem": "nope"})
        assert err
    finally:
        await c.close()
