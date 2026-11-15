#!/usr/bin/env python3
#
#  check_mirror.py
#  snicholls-gpu
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  tests/, demos/ and benchmarks/ mirror include/snicholls/gpu/ (CLAUDE.md "Tests", "Demos"). For every public
#  header <sub>/<name>.hpp:
#
#    tests/<sub>/<name>_test.cpp     listed in tests/CMakeLists.txt
#    demos/<sub>/<name>_demo.cpp     listed in demos/CMakeLists.txt
#
#  A test, demo or benchmark with no header is an error too. Exit status 1 on any finding.
#

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = os.path.join(ROOT, "include", "snicholls", "gpu")


def cmake_text(tree):
    path = os.path.join(ROOT, tree, "CMakeLists.txt")
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


def main():
    problems, headers = [], set()
    for dirpath, _, files in os.walk(INCLUDE):
        for f in sorted(files):
            if f.endswith(".hpp"):
                rel = os.path.relpath(os.path.join(dirpath, f), INCLUDE)[:-4]
                if rel.split(os.sep)[0] != "detail":
                    headers.add(rel)
    for rel in sorted(headers):
        for tree, suffix in (("tests", "_test.cpp"), ("demos", "_demo.cpp")):
            target = rel + suffix
            if not os.path.exists(os.path.join(ROOT, tree, target)):
                problems.append(f"{rel}.hpp: no {tree}/{target}")
            elif target not in cmake_text(tree):
                problems.append(f"{tree}/{target} is not listed in {tree}/CMakeLists.txt")
    for tree, suffix in (("tests", "_test.cpp"), ("demos", "_demo.cpp"), ("benchmarks", "_benchmark.cpp")):
        for dirpath, _, files in os.walk(os.path.join(ROOT, tree)):
            for f in files:
                if f.endswith(suffix):
                    rel = os.path.relpath(os.path.join(dirpath, f), os.path.join(ROOT, tree))[: -len(suffix)]
                    if rel not in headers:
                        problems.append(f"{tree}/{rel}{suffix} has no header")
    for p in problems:
        print("check_mirror: " + p, file=sys.stderr)
    if not problems:
        print(f"check_mirror: ok ({len(headers)} headers, each with a test and a demo)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
