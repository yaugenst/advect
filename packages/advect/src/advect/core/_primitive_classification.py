"""Semantic classifications shared by conformance and evidence tooling."""

from __future__ import annotations

__all__ = ["STRUCTURAL_OPS"]

# Operations with no differentiable user input. This is a semantic
# classification, not a derivative-coverage exemption: any installed rule is
# accounted for separately by the conformance raw-rule matrix.
STRUCTURAL_OPS: dict[str, str] = {
    "advect.input": "trace entry point, not a computation",
    "advect.const": "captured constant, no differentiable input",
    "array.arange": "integer-parameterised creation, no differentiable input",
    "array.empty": "uninitialised creation, no differentiable input",
    "array.eye": "shape-parameterised creation, no differentiable input",
    "array.ones": "shape-parameterised creation, no differentiable input",
    "array.zeros": "shape-parameterised creation, no differentiable input",
    "array.zeros_like": "template input is non-differentiable by contract",
    "array.ones_like": "template input is non-differentiable by contract",
    "array.empty_like": "template input is non-differentiable by contract",
    "array.full_like": "template and static fill value carry no user derivative",
    "array_ext.fft.fftfreq": "frequency grid depends only on integer parameters",
    "array_ext.fft.rfftfreq": "frequency grid depends only on integer parameters",
    "advect.getoutput": "multi-output projection is structural",
    "array.left_shift": "integer bit operation",
    "array.right_shift": "integer bit operation",
}
