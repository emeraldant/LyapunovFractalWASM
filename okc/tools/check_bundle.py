"""Static check of `# @bundle` lines (tools/bundle.py): every definition a bundled
definition references by name, if it is a top-level definition of the same source file,
must be bundled too. Catches NameErrors that otherwise surface only on the judge's GPU.

  python3 okc/tools/check_bundle.py okc/moe_cuda/sim_src.py okc/muse_moe_w4a16/submission_src.py
"""
import ast
import os
import sys

OKC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def top_level(path):
    tree = ast.parse(open(path).read())
    defs = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defs[node.name] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defs[t.id] = node
    return defs


def check(template):
    bad = 0
    own = set(top_level(template))
    for line in open(template):
        if not line.startswith("# @bundle "):
            continue
        rel, *names = line.split()[2:]
        defs = top_level(os.path.join(OKC, rel))
        for name in names:
            if name not in defs:
                print(f"{template}: {name} is not a top-level definition of {rel}")
                bad += 1
                continue
            used = {n.id for n in ast.walk(defs[name]) if isinstance(n, ast.Name)}
            for u in sorted(used & set(defs)):
                if u not in names and u not in own and u != name:
                    print(f"{template}: {name} (from {rel}) uses {u}, which is not bundled")
                    bad += 1
    return bad


if __name__ == "__main__":
    total = sum(check(p) for p in sys.argv[1:])
    print("bundle check:", "ok" if not total else f"{total} problem(s)")
    sys.exit(1 if total else 0)
