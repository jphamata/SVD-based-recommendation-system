"""Render chat templates the way transformers does (differential tier).

usage:
  jinja_render.py            stdin {"templates": {name: src}, "scenarios": [{"name", "vars"}]}
                             stdout {template: {scenario: {"ok": text} | {"error", "raised"}}}
  jinja_render.py snippets   stdin [[name, src], …]  stdout [{"ok"} | {"error", "raised"}, …]

The environment mirrors transformers' `_cached_compile_jinja_template`
(utils/chat_template_utils.py): ImmutableSandboxedEnvironment with
trim_blocks and lstrip_blocks, loop controls, `tojson` as json.dumps with
ensure_ascii=False, `raise_exception`, and `strftime_now` — here on a fixed
clock (2026-10-01 12:00), since a rendered prompt must not depend on when.
"""
import json, sys
from datetime import datetime
import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def strftime_now(fmt):
    return datetime(2026, 10, 1, 12, 0, 0).strftime(fmt)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
env.globals["strftime_now"] = strftime_now


def attempt(f):
    try:
        return {"ok": f()}
    except jinja2.exceptions.TemplateError as e:
        return {"error": str(e), "raised": type(e) is jinja2.exceptions.TemplateError}
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e), "raised": False}


def render_all(req):
    out = {}
    for name, src in req["templates"].items():
        try:
            t = env.from_string(src)
        except Exception as e:
            out[name] = {"__compile__": {"error": str(e), "raised": False}}
            continue
        out[name] = {sc["name"]: attempt(lambda: t.render(**sc["vars"])) for sc in req["scenarios"]}
    return out


def render_snippets(snippets):
    return [attempt(lambda: env.from_string(src).render()) for _name, src in snippets]


if __name__ == "__main__":
    req = json.loads(sys.stdin.read())
    res = render_snippets(req) if sys.argv[1:] == ["snippets"] else render_all(req)
    sys.stdout.write(json.dumps(res, ensure_ascii=False))
