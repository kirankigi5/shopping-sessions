#!/usr/bin/env python3
"""Summarize a samply profile: self and inclusive samples per function. usage: prof.py prof.json.gz [N]"""
import gzip, json, sys, bisect, collections

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
prof = json.load(gzip.open(path))
syms = json.load(open(path.replace(".json.gz", ".json.syms.json")))
strtab = syms["string_table"]
libsyms = {}
for d in syms["data"]:
    st = d["symbol_table"]
    rvas = [e["rva"] for e in st]
    libsyms[d["debug_name"]] = (rvas, st)
libs = prof["libs"]


def name_of(lib_idx, addr):
    lib = libs[lib_idx]
    s = libsyms.get(lib["debugName"])
    if not s:
        return lib["debugName"] + "?"
    rvas, st = s
    i = bisect.bisect_right(rvas, addr) - 1
    if i < 0:
        return "?"
    e = st[i]
    if addr >= e["rva"] + e["size"]:
        return lib["debugName"] + "?"
    return strtab[e["symbol"]]


selfc = collections.Counter()
incl = collections.Counter()
for th in prof["threads"]:
    shared = prof.get("shared", {})
    ft = th["frameTable"]
    funcs = th["funcTable"]
    res = th["resourceTable"]
    st = th["stackTable"]
    samples = th["samples"]
    stacks = samples["stack"]
    cache = {}

    def fname(fi):
        if fi in cache:
            return cache[fi]
        func = ft["func"][fi]
        addr = ft["address"][fi]
        r = funcs["resource"][func]
        lib = res["lib"][r] if r is not None and r >= 0 else None
        n = name_of(lib, addr) if lib is not None and addr >= 0 else "?"
        cache[fi] = n
        return n

    for s in stacks:
        if s is None:
            continue
        seen = set()
        first = True
        while s is not None:
            n = fname(st["frame"][s])
            if first:
                selfc[n] += 1
                first = False
            if n not in seen:
                incl[n] += 1
                seen.add(n)
            s = st["prefix"][s]

import re


def short(n):
    try:
        import subprocess
    except Exception:
        return n
    return n


tot = sum(selfc.values())
print("total samples", tot)
print("== self")
for n, c in selfc.most_common(N):
    print(f"{c:7d} {100*c/tot:5.1f}%  {n[:150]}")
print("== inclusive")
for n, c in incl.most_common(N):
    print(f"{c:7d} {100*c/tot:5.1f}%  {n[:150]}")
