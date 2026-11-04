#!/usr/bin/env python3
"""Codex 1's verbatim-method control as a build step: copy `write_pairs` out of pgens/bh_proto/pgen.hpp, text unchanged, for
tests/hadronic/dump_bounds.cpp. Usage: dump_bounds_extract.py <pgen.hpp> <output .inc>. String and character literals and
comments are skipped when the braces are counted."""
import sys

SIGNATURE = "    void write_pairs(Domain<S, M>& domain) {"


def method(text):
    start = text.index(SIGNATURE)
    i = text.index("{", start)
    depth = 0
    while True:
        c = text[i]
        if text.startswith("//", i):
            i = text.index("\n", i)
            continue
        if text.startswith("/*", i):
            i = text.index("*/", i) + 2
            continue
        if c in "\"'":
            i += 1
            while text[i] != c:
                i += 2 if text[i] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1] + "\n"
        i += 1


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        body = method(f.read())
    with open(sys.argv[2], "w") as f:
        f.write(body)
