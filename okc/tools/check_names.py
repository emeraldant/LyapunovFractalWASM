"""Fail if a built single-file submission references a name it never defines (a forgotten
@bundle entry). Usage: python3 okc/tools/check_names.py <file.py> ..."""
import ast
import builtins
import sys


def unresolved(path):
    tree = ast.parse(open(path).read())
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(used - defined)


if __name__ == "__main__":
    bad = {p: unresolved(p) for p in sys.argv[1:]}
    for p, names in bad.items():
        print(f"{p}: {'unresolved ' + ', '.join(names) if names else 'ok'}")
    sys.exit(1 if any(bad.values()) else 0)
