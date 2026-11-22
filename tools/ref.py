#!/usr/bin/env python3
"""Literal, slow reference implementation of rules.md. Writes CSVs like the fixtures' expected/.
usage: ref.py DATA_DIR OUT_DIR
"""
import csv, glob, json, os, sys, datetime as dt

DATA, OUT = sys.argv[1], sys.argv[2]
DIG = set(b"0123456789")


def lines(path):
    with open(path, "rb") as f:
        data = f.read()
    parts = data.split(b"\n")
    if data.endswith(b"\n"):
        parts = parts[:-1]
    return parts[1:]  # skip header


def parse(line, n):
    if line == b"":
        return None
    fields = line.split(b",")
    if len(fields) != n:
        return None
    out = []
    for f in fields:
        if b'"' in f:
            if len(f) >= 2 and f[0:1] == b'"' and f[-1:] == b'"' and b'"' not in f[1:-1]:
                out.append(f[1:-1])
            else:
                return None
        else:
            out.append(f)
    return out


def alldig(b):
    return len(b) > 0 and all(c in DIG for c in b)


def civil(b):
    # YYYY-MM-DDTHH:MM:SS
    if len(b) != 19 or b[4:5] != b"-" or b[7:8] != b"-" or b[10:11] != b"T" or b[13:14] != b":" or b[16:17] != b":":
        return None
    parts = [b[0:4], b[5:7], b[8:10], b[11:13], b[14:16], b[17:19]]
    if not all(alldig(p) for p in parts):
        return None
    y, mo, d, h, mi, s = (int(p) for p in parts)
    if y < 1:
        # datetime cannot hold year 0; such a date is far outside any window
        return -10**18
    try:
        t = dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return int(t.timestamp()) * 1000


def ts_ms(b):
    if b == b"":
        return None
    if alldig(b):
        v = int(b)
        return v if v < 2**63 else None
    if len(b) == 20 and b[19:20] == b"Z":
        return civil(b[:19])
    if len(b) == 25 and b[19:20] in (b"+", b"-") and b[22:23] == b":":
        base = civil(b[:19])
        if base is None:
            return None
        oh, om = b[20:22], b[23:25]
        if not (alldig(oh) and alldig(om)):
            return None
        oh, om = int(oh), int(om)
        if oh > 23 or om > 59:
            return None
        off = (oh * 60 + om) * 60000
        return base - off if b[19:20] == b"+" else base + off
    return None


man = json.load(open(os.path.join(DATA, "manifest.json")))
WS = ts_ms(man["window_start"].encode())
WE = ts_ms(man["window_end"].encode())
WEEK = 7 * 86400000

# links
parent = {}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for p in sorted(glob.glob(os.path.join(DATA, "links", "*.csv"))):
    for ln in lines(p):
        r = parse(ln, 2)
        if r is None or r[0] == b"" or r[1] == b"":
            continue
        a, b = r
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
groups = {}
for x in list(parent):
    groups.setdefault(find(x), []).append(x)
name = {}
for g in groups.values():
    m = min(g)
    for x in g:
        name[x] = m


def person(u):
    return name.get(u, u)


products = {}
for ln in lines(os.path.join(DATA, "products.csv")):
    r = parse(ln, 2)
    if r:
        products[r[0]] = r[1]
campaigns = {}
for ln in lines(os.path.join(DATA, "campaigns.csv")):
    r = parse(ln, 2)
    if r:
        campaigns[r[0]] = r[1]

# events
best = {}
for p in sorted(glob.glob(os.path.join(DATA, "events", "*.csv"))):
    for ln in lines(p):
        r = parse(ln, 7)
        if r is None or r[0] == b"":
            continue
        ing = ts_ms(r[3])
        k = r[0]
        if k not in best or ing < best[k][0]:
            best[k] = (ing, r)
evs = {}
for k, (ing, r) in best.items():
    eid, user, ts, _, typ, prod, price = r
    if user == b"" or prod == b"":
        continue
    t = ts_ms(ts)
    if t is None or not (WS <= t < WE):
        continue
    if typ not in (b"view", b"cart", b"purchase"):
        continue
    pr = 0
    if typ == b"purchase":
        if not alldig(price):
            continue
        pr = int(price)
        if not (1 <= pr <= 10_000_000):
            continue
    if ing - t > 86400000:
        continue
    cat = products.get(prod, b"unknown") if typ == b"purchase" else None
    evs.setdefault(person(user), []).append((t, typ == b"purchase", pr, cat))

clicks = {}
for p in sorted(glob.glob(os.path.join(DATA, "clicks", "*.csv"))):
    for ln in lines(p):
        r = parse(ln, 4)
        if r is None or r[0] == b"":
            continue
        cid, user, camp, ts = r
        if user == b"":
            continue
        t = ts_ms(ts)
        if t is None or not (WS - WEEK <= t < WE):
            continue
        if camp not in campaigns:
            continue
        clicks.setdefault(person(user), {})[cid] = (t, cid, camp)

ud = {}
cd = {}
for per, es in evs.items():
    es.sort(key=lambda e: e[0])
    sessions = []
    for e in es:
        if sessions and e[0] - sessions[-1][-1][0] <= 1800000:
            sessions[-1].append(e)
        else:
            sessions.append([e])
    pcl = list(clicks.get(per, {}).values())
    for s in sessions:
        start = s[0][0]
        day = dt.datetime.fromtimestamp(start // 1000, dt.timezone.utc).date().isoformat()
        rev = sum(e[2] for e in s if e[1])
        purch = any(e[1] for e in s)
        u = ud.setdefault((per, day), {"sessions": 0, "events": 0, "rev": 0, "cats": {}})
        u["sessions"] += 1
        u["events"] += len(s)
        u["rev"] += rev
        for e in s:
            if e[1]:
                u["cats"][e[3]] = u["cats"].get(e[3], 0) + e[2]
        q = [c for c in pcl if c[0] <= start and start - c[0] <= WEEK]
        if q:
            q.sort(key=lambda c: (-c[0], c[1]))
            camp = q[0][2]
        else:
            camp = b"organic"
        c = cd.setdefault((camp, day), [0, 0, 0, 0])
        c[0] += 1
        c[1] += 1 if purch else 0
        c[2] += rev
        for a in set(c[2] for c in q):
            cc = cd.setdefault((a, day), [0, 0, 0, 0])
            cc[3] += 1

os.makedirs(OUT, exist_ok=True)


def s(b):
    return b.decode("utf-8", "surrogateescape")


with open(os.path.join(OUT, "user_daily.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["user_id", "day", "sessions", "events", "revenue_cents", "top_category"])
    for (per, day), u in sorted(ud.items()):
        top = ""
        if u["cats"]:
            top = s(sorted(u["cats"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
        w.writerow([s(per), day, u["sessions"], u["events"], u["rev"], top])
with open(os.path.join(OUT, "campaign_daily.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["campaign_id", "channel", "day", "sessions", "purchasing_sessions", "revenue_cents", "assisted_sessions"])
    for (camp, day), c in sorted(cd.items()):
        ch = b"organic" if camp == b"organic" else campaigns[camp]
        w.writerow([s(camp), s(ch), day, *c])
