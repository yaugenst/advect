"""Small Pyodide adapter for the documentation playground.

The browser owns presentation only. Every value, derivative, graph, trace,
and report shown by the playground is produced by Advect's public Python API.
"""

from __future__ import annotations

import ast
import json
import math
import time
from importlib.metadata import version
from typing import TYPE_CHECKING, cast, override

import numpy as np

import advect as ad

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import CodeType
    from typing import SupportsFloat

# This browser-facing adapter reports ordinary input errors directly.
# ruff: noqa: INP001, TRY004

_FUNCTIONS = {
    "abs": np.abs,
    "cos": np.cos,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "tanh": np.tanh,
}
_CONSTANTS = {"e": math.e, "pi": math.pi}
_DEF_BUILTINS = {
    "abs": abs,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
    "sum": sum,
    "zip": zip,
}
_EXPR_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)
_DEF_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)
_OP_LABELS = {
    "array.add": "+",
    "array.subtract": "-",
    "array.multiply": "*",
    "array.divide": "/",
    "array.power": "**",
    "array.negative": "neg",
    "array.absolute": "abs",
}

# a value-and-derivative pair, the shape of the playground's jvp display program
_PAIR_OUTPUTS = 2

# (staged value-and-derivative program, dynamic second-derivative callable)
_EVALUATE: tuple[ad.StagedProgram, Callable[[object], object]] | None = None


def _namespace() -> dict[str, object]:
    return {"__builtins__": {}, **_FUNCTIONS, **_CONSTANTS}


# ------------------------------------------------------------- expression mode


def _parse_expression(source: str) -> ast.Expression:
    tree = ast.parse(source, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _EXPR_ALLOWED_NODES):
            raise SyntaxError(f"{type(node).__name__} is not supported in expression mode")
        if isinstance(node, ast.Name) and node.id not in {"x", *_FUNCTIONS, *_CONSTANTS}:
            raise NameError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise SyntaxError("calls must use a supported function name")
            if len(node.args) != 1 or node.keywords:
                raise TypeError(f"{node.func.id}() takes one positional argument here")
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool) or not isinstance(node.value, (int, float))
        ):
            raise TypeError("only real numeric constants are supported")
    if not any(isinstance(node, ast.Name) and node.id == "x" for node in ast.walk(tree)):
        raise ValueError("f must mention x")
    return tree


def _expression_function(tree: ast.Expression) -> Callable[[object], object]:
    code = compile(tree, "<advect-playground>", "eval")
    namespace = _namespace()

    def evaluate(x: object) -> object:
        return eval(code, namespace, {"x": x})  # noqa: S307 - validated expression, local browser

    return evaluate


# --------------------------------------------------------------- function mode


def _def_namespace() -> dict[str, object]:
    """Build function mode's namespace: numpy as ``np`` plus a few builtins."""
    return {"__builtins__": _DEF_BUILTINS, "np": np}


def _function_mode_function(source: str) -> Callable[[object], object]:
    """Compile a ``def f(x):`` body with ordinary Python control flow."""
    module = ast.parse(source, mode="exec")
    for node in ast.walk(module):
        if isinstance(node, _DEF_FORBIDDEN_NODES):
            hint = (
                " (numpy is already available as np)"
                if isinstance(node, (ast.Import, ast.ImportFrom))
                else ""
            )
            raise SyntaxError(f"{type(node).__name__} is not supported in the playground{hint}")
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if [node.name for node in functions].count("f") != 1:
        raise ValueError("define exactly one function named f")
    namespace = _def_namespace()
    exec(compile(module, "<advect-playground>", "exec"), namespace)  # noqa: S102 - runs client-side in the visitor's browser
    function = namespace["f"]
    code = getattr(function, "__code__", None)
    if code is None or code.co_argcount != 1:
        raise TypeError("f must take exactly one positional argument")
    return cast("Callable[[object], object]", function)


# ------------------------------------------------------------------ labelling


def _payload_constant_label(payload: dict[str, object] | None) -> str:
    if payload is None or payload.get("shape") != []:
        return "const"
    try:
        dtype = np.dtype(str(payload["dtype"]))
        value = np.frombuffer(bytes.fromhex(str(payload["data"])), dtype=dtype).item()
    except (KeyError, TypeError, ValueError):
        return "const"
    return f"{value:.5g}" if isinstance(value, float) else str(value)


def _op_label(operation: str, attrs: object) -> str:
    if (
        operation == "array.astype"
        and isinstance(attrs, dict)
        and isinstance(attrs.get("dtype"), dict)
    ):
        return f"astype[{attrs['dtype'].get('value', '?')}]"
    return _OP_LABELS.get(operation, operation.removeprefix("array."))


def _node_label(node: dict[str, object], constants: dict[str, object]) -> str:
    operation = str(node["op"])
    if operation == "advect.input":
        return str(node.get("name") or "x")
    if operation == "advect.const":
        payload = constants.get(str(node["id"]))
        return _payload_constant_label(payload if isinstance(payload, dict) else None)
    return _op_label(operation, node.get("attrs"))


# ----------------------------------------------------------------- graph view


def _reachable(root: int, nodes: dict[int, dict[str, object]]) -> set[int]:
    reached: set[int] = set()
    pending = [root]
    while pending:
        node_id = pending.pop()
        if node_id in reached:
            continue
        reached.add(node_id)
        pending.extend(cast("list[int]", nodes[node_id]["inputs"]))
    return reached


def _descendants(root: int, nodes: dict[int, dict[str, object]]) -> set[int]:
    children: dict[int, list[int]] = {node_id: [] for node_id in nodes}
    for node_id, node in nodes.items():
        for parent in cast("list[int]", node["inputs"]):
            children[parent].append(node_id)
    reached: set[int] = set()
    pending = [root]
    while pending:
        node_id = pending.pop()
        if node_id in reached:
            continue
        reached.add(node_id)
        pending.extend(children[node_id])
    return reached


def _graph_view(graph: dict[str, object], direction: str) -> dict[str, object]:
    raw_nodes = cast("list[dict[str, object]]", graph["nodes"])
    nodes = {cast("int", node["id"]): node for node in raw_nodes}
    outputs = cast("list[int]", graph["outputs"])
    inputs = cast("list[int]", graph["inputs"])
    constants = cast("dict[str, object]", graph["constants"])

    roles: dict[int, str]
    if direction == "vjp":
        # the cotangent seed is the trailing input; everything it touches is
        # the adjoint spine, the rest is primal work the pullback reuses
        adjoint = _descendants(inputs[-1], nodes) if inputs else set()
        roles = {node_id: ("derivative" if node_id in adjoint else "shared") for node_id in nodes}
        for node_id in inputs[:-1]:
            roles[node_id] = "primal"
        headers = ["# grad f"]
    else:
        primal = _reachable(outputs[0], nodes)
        derivative = _reachable(outputs[1], nodes)
        roles = {}
        for node_id in nodes:
            if node_id in primal and node_id in derivative:
                roles[node_id] = "shared"
            else:
                roles[node_id] = "primal" if node_id in primal else "derivative"
        headers = ["# f(x)", "# df/dx"]

    program_lines: list[dict[str, object]] = []
    for node in raw_nodes:
        node_id = cast("int", node["id"])
        label = _node_label(node, constants)
        arguments = ", ".join(f"%{item}" for item in cast("list[object]", node["inputs"]))
        text = f"%{node_id} = {label}" + (f" {arguments}" if arguments else "")
        program_lines.append({"node": node_id, "role": roles[node_id], "text": text})
    program_lines.append({"text": "return " + ", ".join(f"%{item}" for item in outputs)})

    seen: set[int] = set()

    def walk(node_id: int, prefix: str, child_prefix: str, lines: list[dict[str, object]]) -> None:
        shared_reference = node_id in seen
        seen.add(node_id)
        lines.append(
            {
                "node": node_id,
                "pre": prefix,
                "label": _node_label(nodes[node_id], constants),
                "role": roles[node_id],
                "shared": shared_reference,
            }
        )
        if shared_reference:
            return
        node_inputs = cast("list[int]", nodes[node_id]["inputs"])
        for index, input_id in enumerate(node_inputs):
            last = index == len(node_inputs) - 1
            walk(
                input_id,
                f"{child_prefix}{'└─ ' if last else '├─ '}",
                f"{child_prefix}{'   ' if last else '│  '}",
                lines,
            )

    trees = []
    for output in outputs:
        lines: list[dict[str, object]] = []
        walk(output, "", "", lines)
        trees.append(lines)
    return {
        "outputs": outputs,
        "headers": headers,
        "program": program_lines,
        "tree": trees,
        "roles": {str(node_id): role for node_id, role in roles.items()},
    }


# ----------------------------------------------------------------- trace view


def _trace_view(program: ad.StagedProgram, roles: dict[str, str]) -> dict[str, object] | None:
    trace = program.trace
    if trace is None:
        return None
    survivors: dict[int, int] = {}
    rows: list[dict[str, object]] = []
    removed = 0
    merged = 0
    for node, target in zip(trace.nodes, trace.old_to_new, strict=True):
        if node.op == "advect.input":
            label = node.name or "x"
        elif node.op == "advect.const":
            label = "const"
        else:
            label = _op_label(node.op, None)
        arguments = ", ".join(f"%{item}" for item in node.inputs)
        text = f"%{node.id} = {label}" + (f" {arguments}" if arguments else "")
        row: dict[str, object] = {"id": node.id, "text": text}
        if target is None:
            row["status"] = "dead"
            removed += 1
        elif target in survivors:
            row["status"] = "merged"
            row["into"] = survivors[target]
            row["new"] = target
            merged += 1
        else:
            survivors[target] = node.id
            row["status"] = "live"
            row["role"] = roles.get(str(target), "primal")
            row["new"] = target
        rows.append(row)
    return {"rows": rows, "removed": removed, "merged": merged}


# --------------------------------------------------------------- span mapping


def _subexpression_probes(tree: ast.Expression) -> list[tuple[tuple[int, int], CodeType, bool]]:
    """One probe per x-dependent subexpression: (span, code, counts toward folds).

    Bare ``x`` references share the input node trivially; they still get hover
    spans but are not reported as cse folds.
    """
    probes: list[tuple[tuple[int, int], CodeType, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name)):
            continue
        if not any(isinstance(item, ast.Name) and item.id == "x" for item in ast.walk(node)):
            continue
        code = compile(ast.Expression(body=node), "<advect-playground-probe>", "eval")
        end_col_offset = node.end_col_offset if node.end_col_offset is not None else node.col_offset
        probes.append(((node.col_offset, end_col_offset), code, not isinstance(node, ast.Name)))
    return probes


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


class _ProbeMarks(ast.NodeTransformer):
    """Wrap every value-producing subexpression in a recording identity call.

    def-mode sources can't be probed as standalone expressions (subexpressions
    depend on loop variables and local assignments), so instead the function is
    rewritten to report each value as it is computed — one record per dynamic
    occurrence, so loop iterations each surface their own node.
    """

    def __init__(self, offsets: list[int]) -> None:
        self.spans: list[tuple[int, int]] = []
        self.folds: list[bool] = []
        self._offsets = offsets

    def _mark(self, node: ast.expr) -> ast.expr:
        self.generic_visit(node)
        end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
        end_col = node.end_col_offset if node.end_col_offset is not None else node.col_offset
        index = len(self.spans)
        self.spans.append(
            (
                self._offsets[node.lineno - 1] + node.col_offset,
                self._offsets[end_lineno - 1] + end_col,
            )
        )
        self.folds.append(not isinstance(node, ast.Name))
        return ast.Call(
            func=ast.Name(id="_advect_probe", ctx=ast.Load()),
            args=[ast.Constant(value=index), node],
            keywords=[],
        )

    @override
    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        return self._mark(node)

    @override
    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        return self._mark(node)

    @override
    def visit_Call(self, node: ast.Call) -> ast.expr:
        return self._mark(node)

    @override
    def visit_Name(self, node: ast.Name) -> ast.expr:
        if isinstance(node.ctx, ast.Load):
            return self._mark(node)
        return node


def _structural_hashes(graph: dict[str, object]) -> dict[int, object]:
    """Backend-independent structural digest per node, for cross-graph matching."""
    nodes = {
        cast("int", node["id"]): node for node in cast("list[dict[str, object]]", graph["nodes"])
    }
    constants = cast("dict[str, object]", graph["constants"])
    memo: dict[int, object] = {}

    def visit(node_id: int) -> object:
        if node_id in memo:
            return memo[node_id]
        node = nodes[node_id]
        operation = str(node["op"])
        if operation == "advect.input":
            digest: object = ("input", str(node.get("name")))
        elif operation == "advect.const":
            payload = constants.get(str(node_id))
            record = payload if isinstance(payload, dict) else {}
            digest = (
                "const",
                str(record.get("dtype")),
                str(record.get("data")),
                str(record.get("shape")),
            )
        else:
            attrs = node.get("attrs")
            semantic = {
                key: value
                for key, value in (attrs.items() if isinstance(attrs, dict) else ())
                if not key.startswith("_")
            }
            digest = (
                operation,
                json.dumps(semantic, sort_keys=True, default=str),
                tuple(visit(item) for item in cast("list[int]", node["inputs"])),
            )
        memo[node_id] = digest
        return digest

    return {node_id: visit(node_id) for node_id in nodes}


def _match_spans(
    pairs: list[tuple[tuple[int, int], bool]],
    probe_outputs: list[int],
    probe_graph: dict[str, object],
    display_graph: dict[str, object],
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, int]]:
    """Match probe program outputs to display nodes by structural digest.

    A display node claimed by several source spans is one node computed for
    many source sites — cse made visible. ``folds`` carries that count for
    nodes that are the unique owner of their digest.
    """
    probe_hashes = _structural_hashes(probe_graph)
    display_by_hash: dict[object, list[int]] = {}
    for node_id, digest in _structural_hashes(display_graph).items():
        display_by_hash.setdefault(digest, []).append(node_id)
    spans: dict[str, list[tuple[int, int]]] = {}
    fold_counts: dict[str, int] = {}
    for (span, foldable), output in zip(pairs, probe_outputs, strict=True):
        owners = display_by_hash.get(probe_hashes[output], [])
        for node_id in owners:
            entry = spans.setdefault(str(node_id), [])
            if span not in entry:
                entry.append(span)
                if foldable and len(owners) == 1:
                    fold_counts[str(node_id)] = fold_counts.get(str(node_id), 0) + 1
    folds = {node_id: count for node_id, count in fold_counts.items() if count > 1}
    return spans, folds


def _source_spans(
    tree: ast.Expression, display_graph: dict[str, object]
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, int]]:
    """Map display nodes to the expression source spans that produced them.

    A throwaway probe program is staged whose outputs are every x-dependent
    subexpression; its output nodes are matched to the display graph by
    structural digest.
    """
    probes = _subexpression_probes(tree)
    if not probes:
        return {}, {}
    codes = [code for _, code, _ in probes]
    namespace = _namespace()

    def probe_values(x: object) -> tuple[object, ...]:
        return tuple(
            eval(code, namespace, {"x": x})  # noqa: S307 - validated
            for code in codes
        )

    probe_program = cast(
        "ad.StagedProgram",
        ad.stage(probe_values, 0.0),
    )
    probe_artifact = cast("dict[str, object]", probe_program.to_dict()["program"])
    probe_graph = cast("dict[str, object]", probe_artifact["graph"])
    pairs = [(span, foldable) for span, _, foldable in probes]
    return _match_spans(
        pairs, cast("list[int]", probe_graph["outputs"]), probe_graph, display_graph
    )


def _def_source_spans(
    source: str, display_graph: dict[str, object]
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, int]]:
    """Map display nodes to ``def f`` source spans.

    The already-validated module is re-parsed with ``_ProbeMarks`` so a single
    trace records the tracer built at every source site, loop iterations
    included. Those tracers become the outputs of a throwaway probe program;
    matching then works exactly as in expression mode.
    """
    module = ast.parse(source, mode="exec")
    marks = _ProbeMarks(_line_offsets(source))
    marked = ast.fix_missing_locations(marks.visit(module))
    records: list[tuple[int, object]] = []

    def probe(index: int, value: object) -> object:
        records.append((index, value))
        return value

    namespace = _def_namespace()
    namespace["_advect_probe"] = probe
    exec(compile(marked, "<advect-playground-probe>", "exec"), namespace)  # noqa: S102 - source validated by _function_mode_function, client-side
    function = cast("Callable[[object], object]", namespace["f"])
    pairs: list[tuple[tuple[int, int], bool]] = []

    def probed(x: object) -> object:
        records.clear()
        result = function(x)
        pairs.clear()
        values: list[object] = []
        for index, value in records:
            if isinstance(value, type(x)):
                pairs.append((marks.spans[index], marks.folds[index]))
                values.append(value)
        return (result, *values)

    probe_program = cast("ad.StagedProgram", ad.stage(probed, 0.0))
    probe_artifact = cast("dict[str, object]", probe_program.to_dict()["program"])
    probe_graph = cast("dict[str, object]", probe_artifact["graph"])
    outputs = cast("list[int]", probe_graph["outputs"])
    return _match_spans(pairs, outputs[1:], probe_graph, display_graph)


# -------------------------------------------------------------------- tracing


def _finite_list(values: object) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values)]


def _report_payload(
    program: ad.StagedProgram,
    eval_seconds: float | None,
    samples: int,
) -> dict[str, object]:
    report = program.optimization
    payload: dict[str, object] = {
        "nodes_before": report.nodes_before,
        "nodes_after": report.nodes_after,
        "compile_seconds": program.compile_seconds,
        "passes": [
            {
                "name": item.name,
                "nodes_before": item.nodes_before,
                "nodes_after": item.nodes_after,
            }
            for item in report.passes
        ],
    }
    if eval_seconds is not None:
        payload["eval_seconds"] = eval_seconds
        payload["samples"] = samples
    return payload


def playground_trace_json(source: str, mode: str = "expr", direction: str = "jvp") -> str:
    """Trace f, stage its derivative programs, and return the full view state."""
    global _EVALUATE  # noqa: PLW0603

    expression_tree: ast.Expression | None = None
    if mode == "def":
        function = _function_mode_function(source)
    else:
        expression_tree = _parse_expression(source)
        function = _expression_function(expression_tree)

    derivative = ad.jvp(function)
    example_input = 0.0

    def first_derivative(x: object) -> object:
        return derivative(x, tangents=1.0)

    jvp_program = cast("ad.StagedProgram", ad.stage(first_derivative, example_input))
    # staging rejects higher-order autodiff; the curvature readout runs the
    # dynamic transform instead — advect's define-by-run path, per evaluation
    _EVALUATE = (jvp_program, ad.hessian(function))

    if direction == "vjp":
        primal_program = cast("ad.StagedProgram", ad.stage(function, example_input))
        display_program = ad.vjp_program(primal_program)
    else:
        display_program = jvp_program

    xs = np.linspace(-4.0, 4.0, 112)
    values: list[object] = []
    derivatives: list[object] = []
    eval_started = time.perf_counter()
    with np.errstate(all="ignore"):
        for x in xs:
            value, derivative = jvp_program(float(x))
            values.append(value)
            derivatives.append(derivative)
    eval_seconds = time.perf_counter() - eval_started

    artifact = cast("dict[str, object]", display_program.to_dict()["program"])
    graph = cast("dict[str, object]", artifact["graph"])
    view = _graph_view(graph, direction)
    roles = cast("dict[str, str]", view["roles"])
    view["trace"] = _trace_view(display_program, roles)
    spans: dict[str, list[tuple[int, int]]] = {}
    folds: dict[str, int] = {}
    try:
        if expression_tree is not None:
            spans, folds = _source_spans(expression_tree, graph)
        else:
            spans, folds = _def_source_spans(source, graph)
    except Exception:  # noqa: BLE001 - highlighting is optional presentation
        spans.clear()
        folds.clear()
    view["spans"] = spans
    view["folds"] = folds

    return json.dumps(
        {
            "graph": view,
            "direction": direction,
            "mode": mode,
            "series": {
                "values": _finite_list(values),
                "derivatives": _finite_list(derivatives),
            },
            "report": _report_payload(display_program, eval_seconds, len(xs)),
            "runtime": {
                "advect": version("advect"),
                "numpy": np.__version__,
            },
        },
        allow_nan=False,
    )


def playground_evaluate_json(x: SupportsFloat) -> str:
    """Evaluate f and f' via the staged program, and f'' dynamically."""
    if _EVALUATE is None:
        raise RuntimeError("trace an expression before evaluating it")
    jvp_program, hessian = _EVALUATE

    def to_json(item: object) -> float | None:
        value = float(cast("SupportsFloat", item))
        return value if math.isfinite(value) else None

    point = float(x)
    with np.errstate(all="ignore"):
        value, derivative = jvp_program(point)
        try:
            second = to_json(hessian(point))
        except Exception:  # noqa: BLE001 - curvature is an optional readout
            second = None
    return json.dumps([to_json(value), to_json(derivative), second])


def playground_names_json() -> str:
    """Autocomplete candidates for both editors, drawn from the real namespaces."""
    ufuncs = {name for name in dir(np) if isinstance(getattr(np, name), np.ufunc)}
    np_extras = {"asarray", "clip", "e", "inf", "linspace", "nan", "pi", "where"}
    keywords = {"def", "elif", "else", "for", "if", "in", "lambda", "return", "while"}
    return json.dumps(
        {
            "expr": sorted({"x", *_FUNCTIONS, *_CONSTANTS}),
            "def": {
                "np": sorted(ufuncs | np_extras),
                "plain": sorted({"np", "x", *_DEF_BUILTINS, *keywords}),
            },
        }
    )


def playground_artifact_json() -> str:
    """Serialize the current first-derivative program as a durable artifact."""
    if _EVALUATE is None:
        raise RuntimeError("trace an expression before saving it")
    return json.dumps(_EVALUATE[0].to_dict())


def playground_load_json(text: str) -> str:
    """Load a durable artifact for inspection and return its graph view."""
    program = ad.StagedProgram.from_dict(json.loads(text))
    artifact = cast("dict[str, object]", program.to_dict()["program"])
    graph = cast("dict[str, object]", artifact["graph"])
    outputs = cast("list[int]", graph["outputs"])
    direction = "jvp" if len(outputs) == _PAIR_OUTPUTS else "vjp"
    view = _graph_view(graph, direction)
    view["trace"] = None
    empty_spans: dict[str, list[tuple[int, int]]] = {}
    empty_folds: dict[str, int] = {}
    view["spans"] = empty_spans
    view["folds"] = empty_folds
    return json.dumps(
        {
            "graph": view,
            "direction": direction,
            "inputs": len(cast("list[int]", graph["inputs"])),
            "outputs": len(outputs),
            "report": _report_payload(program, None, 0),
        },
        allow_nan=False,
    )
