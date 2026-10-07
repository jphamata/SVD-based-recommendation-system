"""Python's verdict on vapor's pattern/format grammars.

stdin: {"patterns": [{"pattern", "strings"}], "formats": [{"format", "strings"}]}
stdout: the same lists of booleans — re.search for patterns (ECMA-262
semantics restored where Python's differ: `.`, `$`, \\d \\w \\s are
rewritten to their ECMA classes, named groups to Python's spelling), and
for formats the standard library's own parsers: datetime (with RFC 3339's
leap second), ipaddress, uuid, email.headerregistry, and RFC 1123 for
host names.
"""
import datetime, ipaddress, json, re, sys, uuid
from email.headerregistry import Address

ECMA = {"d": "0-9", "w": "A-Za-z0-9_",
        "s": "\\t\\n\\x0b\\x0c\\r \\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff"}


def ecma(p):
    out, i = [], 0
    while i < len(p):
        c = p[i]
        if c == "\\" and i + 1 < len(p):
            e = p[i + 1]
            if e.lower() in ECMA:
                out.append(("[" if e.islower() else "[^") + ECMA[e.lower()] + "]")
            else:
                out.append(p[i:i + 2])
            i += 2
        elif c == "[":
            # a class: positive members kept, negated escapes (\S \D \W) as
            # alternatives beside it — Python has no ECMA spelling for them inside
            j, neg, items, negs = i + 1, False, [], []
            if p[j:j + 1] == "^":
                neg, j = True, j + 1
            while p[j] != "]":
                if p[j] == "\\":
                    e = p[j + 1]
                    if e.lower() in ECMA:
                        (items.append(ECMA[e]) if e.islower() else negs.append(ECMA[e.lower()]))
                    else:
                        items.append(p[j:j + 2])
                    j += 2
                else:
                    items.append("\\]" if p[j] == "]" else p[j]); j += 1
            assert not (neg and negs), "a negated class with a negated escape"
            body = "".join(items)
            alts = ([("[^" if neg else "[") + body + "]"] if body or neg else []) + ["[^" + n + "]" for n in negs]
            out.append("(?:" + "|".join(alts) + ")" if alts else "(?!)")
            i = j + 1
        elif c == ".":
            out.append("[^\\n\\r\\u2028\\u2029]"); i += 1
        elif c == "$":
            out.append("\\Z"); i += 1
        elif p.startswith("(?<", i) and p[i + 3:i + 4] not in ("=", "!"):
            out.append("(?P<"); i += 3
        else:
            out.append(c); i += 1
    return "".join(out)


def rfc3339_time(s):
    m = re.fullmatch(r"(\d\d):(\d\d):(\d\d)(\.\d+)?([Zz]|[+-](\d\d):(\d\d))", s)
    if not m:
        return False
    h, mi, se = int(m[1]), int(m[2]), int(m[3])
    if m[6] is not None and (int(m[6]) > 23 or int(m[7]) > 59):
        return False
    return h <= 23 and mi <= 59 and se <= 60


def full_date(s):
    if not re.fullmatch(r"\d{4}-\d\d-\d\d", s):
        return False
    # RFC 3339 admits year 0000 (Python's datetime starts at 1): the
    # proleptic Gregorian calendar, written out
    y, m, d = int(s[:4]), int(s[5:7]), int(s[8:10])
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if y >= 1:
        assert (1 <= m <= 12 and 1 <= d <= days[m - 1]) == _datetime_ok(y, m, d)
    return 1 <= m <= 12 and 1 <= d <= days[m - 1]


def _datetime_ok(y, m, d):
    try:
        datetime.date(y, m, d)
        return True
    except ValueError:
        return False


def email(s):
    try:
        a = Address(addr_spec=s)
        return a.addr_spec == s and "." in a.domain
    except Exception:
        return False


def hostname(s):
    labels = s.split(".")
    return len(s) <= 253 and all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", l) for l in labels)


def ipv4(s):
    try:
        ipaddress.IPv4Address(s)
        return True
    except ValueError:
        return False


def uuid_ok(s):
    try:
        return str(uuid.UUID(s)) == s.lower() and len(s) == 36
    except ValueError:
        return False


FORMATS = {"date": full_date, "time": rfc3339_time,
           "date-time": lambda s: len(s) > 11 and s[10] in "Tt" and full_date(s[:10]) and rfc3339_time(s[11:]),
           "ipv4": ipv4, "uuid": uuid_ok, "email": email, "hostname": hostname}

req = json.load(sys.stdin)
out = {"patterns": [[re.search(ecma(c["pattern"]), s) is not None for s in c["strings"]] for c in req["patterns"]],
       "formats": [[FORMATS[c["format"]](s) for s in c["strings"]] for c in req["formats"]]}
json.dump(out, sys.stdout)
