"""The official MCP Python SDK as a client of `mix vapor.mcp`.

usage: mcp_client.py PROJECT_DIR STUDIO_DIR
Prints one JSON object: the tool names, and per call its isError, text and
structured content (images reduced to their MIME type and size).
"""
import asyncio, base64, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

project, studio = sys.argv[1], sys.argv[2]
graph = {"nodes": {
    "1": {"type": "image.solid", "params": {"width": 32, "height": 24, "red": 0.9, "green": 0.3, "blue": 0.1}},
    "2": {"type": "image.blur", "params": {"sigma": 1.5}, "inputs": {"image": ["1", "image"]}},
    "3": {"type": "audio.tone", "params": {"frequency": 440.0, "seconds": 0.25}},
    "4": {"type": "studio.output", "params": {"name": "picture"}, "inputs": {"value": ["2", "image"]}},
    "5": {"type": "studio.output", "params": {"name": "beep"}, "inputs": {"value": ["3", "audio"]}}}}
edited = json.loads(json.dumps(graph))
edited["nodes"]["3"]["params"]["frequency"] = 880.0
bad = {"nodes": {"1": {"type": "image.solid"}, "2": {"type": "audio.gain", "inputs": {"audio": ["1", "image"]}}}}


def a(o, snake, camel):
    # the SDK's attribute names changed from camelCase to snake_case
    return getattr(o, snake) if hasattr(o, snake) else getattr(o, camel)


def summary(r):
    content = []
    for c in r.content:
        if c.type == "text":
            content.append({"type": "text", "text": c.text})
        elif c.type == "image":
            content.append({"type": "image", "mime": a(c, "mime_type", "mimeType"), "bytes": len(base64.b64decode(c.data))})
    return {"isError": a(r, "is_error", "isError"), "content": content, "structured": a(r, "structured_content", "structuredContent")}


async def main():
    env = dict(os.environ, MIX_ENV=os.environ.get("MIX_ENV", "test"))
    params = StdioServerParameters(command="mix", args=["vapor.mcp", "--dir", studio], cwd=project, env=env)
    out = {}
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            out["server"] = a(init, "server_info", "serverInfo").name
            out["tools"] = sorted(t.name for t in (await s.list_tools()).tools)
            out["catalogue"] = summary(await s.call_tool("studio_catalogue", {"category": "diffusion"}))
            out["invalid"] = summary(await s.call_tool("studio_validate", {"graph": bad}))
            out["valid"] = summary(await s.call_tool("studio_validate", {"graph": graph}))
            out["run1"] = summary(await s.call_tool("studio_run", {"graph": graph}))
            out["run2"] = summary(await s.call_tool("studio_run", {"graph": edited}))
            root = out["run1"]["structured"]["root"]
            out["verify"] = summary(await s.call_tool("studio_verify", {"graph": graph, "root": root}))
            out["verify_wrong"] = summary(await s.call_tool("studio_verify", {"graph": edited, "root": root}))
            out["search"] = summary(await s.call_tool("context_search", {"query": "marching tetrahedra watertight", "paths": ["notes.md", "other.md"], "k": 2}))
            out["escape"] = summary(await s.call_tool("context_search", {"query": "x", "paths": ["../../etc/passwd"]}))
    print(json.dumps(out))

asyncio.run(main())
