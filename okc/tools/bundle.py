"""Build a single-file submission from a template and other submissions' definitions.

A line `# @bundle <path relative to okc/> name1 name2 ...` in the template is replaced by
those top-level definitions (functions with their decorators, classes, or assignments),
copied verbatim from that file. Usage: python tools/bundle.py <template> <output>."""
import ast
import os
import sys

OKC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def definitions(path):
    src = open(path).read()
    lines = src.splitlines(keepends=True)
    out = {}
    for node in ast.parse(src).body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        for name in names:
            out[name] = "".join(lines[start:node.end_lineno])
    return out


def main(template, output):
    result = []
    for line in open(template):
        if line.startswith("# @bundle "):
            rel, *names = line.split()[2:]
            defs = definitions(os.path.join(OKC, rel))
            for name in names:
                result.append(defs[name].rstrip("\n") + "\n\n\n")
        else:
            result.append(line)
    open(output, "w").write("".join(result).rstrip("\n") + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
