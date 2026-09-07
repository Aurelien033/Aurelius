"""aurelius_cli/pipeline_commands.py

CLI for streaming JSONL data transformations using Pipeline.
"""

from __future__ import annotations

import argparse
import ast
import json
<<<<<<< Updated upstream
import operator
=======
import operator as _op
>>>>>>> Stashed changes
import sys
from collections.abc import Callable
from typing import Any

from aurelius_cli.pipeline_processor import Pipeline

# ---------------------------------------------------------------------------
# Safe expression evaluator — no eval/exec.
# Traverses the AST directly so untrusted input cannot reach Python builtins
# via the __class__.__bases__[0].__subclasses__() class-hierarchy bypass.
# ---------------------------------------------------------------------------

_BINOP_MAP: dict[type, Callable] = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod,
    ast.Pow: _op.pow, ast.BitAnd: _op.and_, ast.BitOr: _op.or_,
}
_UNARYOP_MAP: dict[type, Callable] = {
    ast.USub: _op.neg, ast.UAdd: _op.pos, ast.Not: _op.not_,
}
_CMPOP_MAP: dict[type, Callable] = {
    ast.Eq: _op.eq, ast.NotEq: _op.ne,
    ast.Lt: _op.lt, ast.LtE: _op.le,
    ast.Gt: _op.gt, ast.GtE: _op.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_ALLOWED_METHODS = frozenset({
    "get", "keys", "values", "items", "upper", "lower", "strip", "split",
    "startswith", "endswith", "replace", "join", "format", "count",
    "index", "find", "append", "extend", "pop",
})
_ALLOWED_BUILTINS = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "set": set, "sum": sum, "min": min,
    "max": max, "abs": abs, "round": round, "sorted": sorted,
    "any": any, "all": all, "isinstance": isinstance,
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


<<<<<<< Updated upstream
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_UNARY_OPS = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}
_SAFE_FUNCTIONS = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "str": str,
}
_SAFE_METHODS = {
    "count",
    "endswith",
    "find",
    "get",
    "index",
    "isalnum",
    "isalpha",
    "isdigit",
    "islower",
    "isspace",
    "istitle",
    "isupper",
    "join",
    "lower",
    "replace",
    "split",
    "startswith",
    "strip",
    "title",
    "upper",
}


class _SafeExpression:
    """Small expression evaluator for JSONL pipeline transforms.

    This intentionally supports data-shaping expressions over the single
    variable ``x`` and rejects imports, comprehensions, attribute traversal,
    assignment, and arbitrary function calls. It replaces the previous CLI
    ``eval`` path while preserving common examples such as ``x["age"] > 18``
    and ``x["name"].upper()``.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        tree = ast.parse(source, mode="eval")
        body = tree.body
        if isinstance(body, ast.Lambda):
            args = body.args.args
            if len(args) != 1 or args[0].arg != "x":
                raise ValueError("lambda expressions must accept exactly one argument named 'x'")
            body = body.body
        self.body = body

    def __call__(self, x: object) -> object:
        return self._eval(self.body, {"x": x})

    def _eval(self, node: ast.AST, env: dict[str, object]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "x":
                return env["x"]
            raise ValueError(f"unknown name {node.id!r}; only 'x' is allowed")
        if isinstance(node, ast.List):
            return [self._eval(elt, env) for elt in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(elt, env) for elt in node.elts)
        if isinstance(node, ast.Dict):
            return {
                self._eval(key, env): self._eval(value, env)
                for key, value in zip(node.keys, node.values, strict=True)
            }
        if isinstance(node, ast.Subscript):
            return self._eval(node.value, env)[self._eval_slice(node.slice, env)]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](self._eval(node.left, env), self._eval(node.right, env))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](self._eval(node.operand, env))
        if isinstance(node, ast.BoolOp):
            values = [self._eval(value, env) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, env)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = self._eval(comparator, env)
                op_fn = _CMP_OPS.get(type(op))
                if op_fn is None:
                    raise ValueError(f"unsupported comparison operator {type(op).__name__}")
                if not op_fn(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._eval(node.body if self._eval(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Call):
            return self._eval_call(node, env)
        raise ValueError(f"unsupported expression node {type(node).__name__}")

    def _eval_slice(self, node: ast.AST, env: dict[str, object]) -> Any:
        if isinstance(node, ast.Slice):
            lower = self._eval(node.lower, env) if node.lower is not None else None
            upper = self._eval(node.upper, env) if node.upper is not None else None
            step = self._eval(node.step, env) if node.step is not None else None
            return slice(lower, upper, step)
        return self._eval(node, env)

    def _eval_call(self, node: ast.Call, env: dict[str, object]) -> object:
        if node.keywords:
            raise ValueError("keyword arguments are not supported in pipeline expressions")
        args = [self._eval(arg, env) for arg in node.args]
        if isinstance(node.func, ast.Name):
            fn = _SAFE_FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ValueError(f"function {node.func.id!r} is not allowed")
            return fn(*args)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("_") or node.func.attr not in _SAFE_METHODS:
                raise ValueError(f"method {node.func.attr!r} is not allowed")
            target = self._eval(node.func.value, env)
            return getattr(target, node.func.attr)(*args)
        raise ValueError("unsupported callable in pipeline expression")


def _compile_expr(expr: str) -> Callable[[object], object]:
    """Compile a user-supplied data expression into a safe callable."""
    return _SafeExpression(expr.strip())


=======
>>>>>>> Stashed changes
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
