"""aurelius_cli/pipeline_commands.py

CLI for streaming JSONL data transformations using Pipeline.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator as _op
import sys
from collections.abc import Callable

from aurelius_cli.pipeline_processor import Pipeline

# ---------------------------------------------------------------------------
# Safe expression evaluator — no eval/exec.
# Traverses the AST directly so untrusted input cannot reach Python builtins
# via the __class__.__bases__[0].__subclasses__() class-hierarchy bypass.
# ---------------------------------------------------------------------------

_BINOP_MAP: dict[type, Callable] = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
    ast.BitAnd: _op.and_,
    ast.BitOr: _op.or_,
}
_UNARYOP_MAP: dict[type, Callable] = {
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
    ast.Not: _op.not_,
}
_CMPOP_MAP: dict[type, Callable] = {
    ast.Eq: _op.eq,
    ast.NotEq: _op.ne,
    ast.Lt: _op.lt,
    ast.LtE: _op.le,
    ast.Gt: _op.gt,
    ast.GtE: _op.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_ALLOWED_METHODS = frozenset(
    {
        "get",
        "keys",
        "values",
        "items",
        "upper",
        "lower",
        "strip",
        "split",
        "startswith",
        "endswith",
        "replace",
        "join",
        "format",
        "count",
        "index",
        "find",
        "append",
        "extend",
        "pop",
    }
)
_ALLOWED_BUILTINS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "any": any,
    "all": all,
    "isinstance": isinstance,
}


def _walk(node: ast.AST, x: object) -> object:
    """Recursively evaluate a safe AST expression with 'x' as the bound variable."""
    if isinstance(node, ast.Expression):
        return _walk(node.body, x)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "x":
            return x
        if node.id in _ALLOWED_BUILTINS:
            return _ALLOWED_BUILTINS[node.id]
        raise ValueError(f"Unknown name {node.id!r}")
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("__"):
            raise ValueError(f"Dunder attribute access not allowed: {node.attr!r}")
        if node.attr not in _ALLOWED_METHODS:
            raise ValueError(f"Method not allowed: {node.attr!r}")
        obj = _walk(node.value, x)
        return getattr(obj, node.attr)
    if isinstance(node, ast.Subscript):
        obj = _walk(node.value, x)
        key = _walk(node.slice, x)
        return obj[key]  # type: ignore[index]
    if isinstance(node, ast.Index):  # Python 3.8 compat
        return _walk(node.value, x)  # type: ignore[attr-defined]
    if isinstance(node, ast.BinOp):
        op = _BINOP_MAP.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return op(_walk(node.left, x), _walk(node.right, x))
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOP_MAP.get(type(node.op))
        if op is None:
            raise ValueError(f"Unary operator not allowed: {type(node.op).__name__}")
        return op(_walk(node.operand, x))
    if isinstance(node, ast.BoolOp):
        values = [_walk(v, x) for v in node.values]
        if isinstance(node.op, ast.And):
            result: object = True
            for v in values:
                result = result and v
            return result
        result = False
        for v in values:
            result = result or v
        return result
    if isinstance(node, ast.Compare):
        left = _walk(node.left, x)
        for op_node, comp_node in zip(node.ops, node.comparators):
            op = _CMPOP_MAP.get(type(op_node))
            if op is None:
                raise ValueError(f"Compare op not allowed: {type(op_node).__name__}")
            right = _walk(comp_node, x)
            if not op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        func = _walk(node.func, x)
        args = [_walk(a, x) for a in node.args]
        kwargs = {kw.arg: _walk(kw.value, x) for kw in node.keywords if kw.arg}
        return func(*args, **kwargs)
    if isinstance(node, ast.IfExp):
        cond = _walk(node.test, x)
        return _walk(node.body, x) if cond else _walk(node.orelse, x)
    if isinstance(node, ast.List):
        return [_walk(e, x) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_walk(e, x) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_walk(k, x): _walk(v, x) for k, v in zip(node.keys, node.values)}
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _compile_expr(expr: str) -> Callable[[object], object]:
    """Compile a user-supplied expression into a safe callable.

    Parses the expression with the standard AST parser, then evaluates nodes
    directly via _walk() — no eval/exec, so sandbox bypass via __builtins__
    injection is not possible.

    Accepts:
    - ``lambda x: ...`` form (the lambda body is extracted and walked)
    - Bare expression ``x["age"] > 18`` (x is the record)
    """
    code = expr.strip()
    if code.startswith("lambda "):
        source = code
    else:
        source = f"lambda x: {code}"

    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    # tree.body is a Lambda node; extract its body for walking
    body = tree.body
    if not isinstance(body, ast.Lambda):
        raise ValueError("Expected a lambda expression")
    expr_body = body.body

    def _fn(x: object) -> object:
        return _walk(expr_body, x)

    return _fn


def build_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add ``aurelius pipeline`` command to the top-level CLI."""
    parser = subparsers.add_parser(
        "pipeline",
        help="Stream JSONL transformations (filter/map/sort/head/tail/dedup)",
    )
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        help='Keep items where expression is true; expression uses "x" (e.g. x["age"] > 18)',
    )
    parser.add_argument(
        "--map",
        dest="map_expr",
        help='Transform each item; expression uses "x" (e.g. x["name"].upper())',
    )
    parser.add_argument(
        "--sort",
        dest="sort_key_expr",
        help='Sort items by key expression using "x" (e.g. x["score"])',
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        default=False,
        help="Sort in descending order (requires --sort)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Emit only the first N items",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=None,
        help="Emit only the last N items",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        default=False,
        help="Remove duplicate items (keeps first occurrence, preserves order)",
    )
    parser.add_argument(
        "--dedup-key",
        dest="dedup_key_expr",
        help='Deduplicate by key expression using "x" (e.g. x["id"])',
    )
    parser.set_defaults(func=handle_pipeline)


def handle_pipeline(args: argparse.Namespace) -> int:
    """Read JSON lines from stdin, apply transformations, write JSON lines to stdout."""
    filter_fn = _compile_expr(args.filter_expr) if args.filter_expr else None
    map_fn = _compile_expr(args.map_expr) if args.map_expr else None
    sort_key_fn = _compile_expr(args.sort_key_expr) if args.sort_key_expr else None
    dedup_key_fn = _compile_expr(args.dedup_key_expr) if args.dedup_key_expr else None

    try:
        items = [json.loads(line) for line in sys.stdin if line.strip()]
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON on stdin — {exc}", file=sys.stderr)
        return 1

    p = Pipeline(items)

    try:
        if filter_fn:
            p = p.filter(filter_fn)
        if map_fn:
            p = p.map(map_fn)
        if sort_key_fn:
            p = p.sort(key=sort_key_fn, reverse=args.reverse)
        if args.head is not None:
            if args.head < 0:
                print("error: --head must be non-negative", file=sys.stderr)
                return 1
            p = p.head(args.head)
        if args.tail is not None:
            if args.tail < 0:
                print("error: --tail must be non-negative", file=sys.stderr)
                return 1
            p = p.tail(args.tail)
    except Exception as exc:
        print(f"error: pipeline stage failed — {exc}", file=sys.stderr)
        return 1

    results = list(p)

    if args.dedup or args.dedup_key_expr:
        if args.dedup_key_expr:
            seen = set()
            uniq = []
            for item in results:
                k = dedup_key_fn(item)
                if k not in seen:
                    seen.add(k)
                    uniq.append(item)
            results = uniq
        else:
            seen = set()
            uniq = []
            for item in results:
                key = json.dumps(item, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    uniq.append(item)
            results = uniq

    for item in results:
        print(json.dumps(item, ensure_ascii=False))

    return 0
