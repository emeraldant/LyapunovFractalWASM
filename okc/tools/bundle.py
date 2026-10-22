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
    seen = {}
    for line in open(template):
        if line.startswith("# @bundle "):
            rel, *names = line.split()[2:]
            defs = definitions(os.path.join(OKC, rel))
            for name in names:
                if name in seen and seen[name] != defs[name]:
                    raise SystemExit(f"bundle: {name} from {rel} differs from an earlier definition")
                seen[name] = defs[name]
                result.append(defs[name].rstrip("\n") + "\n\n\n")
        else:
            result.append(line)
    text = "".join(result).rstrip("\n") + "\n"
    # Top-level names defined twice with different source would silently shadow each other.
    final = {}
    for node in ast.parse(text).body:
        names = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else [
            t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
        seg = ast.get_source_segment(text, node)
        for name in names:
            if name in final and final[name] != seg:
                raise SystemExit(f"bundle: top-level {name} is defined twice with different source")
            final[name] = seg
    open(output, "w").write(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
