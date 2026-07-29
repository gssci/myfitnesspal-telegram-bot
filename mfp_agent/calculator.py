from __future__ import annotations

import ast
import json
import math
import operator
from collections.abc import Callable

from langchain_core.tools import tool

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression(expression: str) -> int | float:
    """Evaluate a bounded arithmetic expression without executing Python code."""
    if not expression.strip() or len(expression) > 200:
        raise ValueError("Expression must contain between 1 and 200 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid arithmetic expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError("Expression is too complex")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent is outside the safe range")
            try:
                result = _BINARY_OPERATORS[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise ValueError("Division by zero") from exc
            if not math.isfinite(result) or abs(result) > 1e15:
                raise ValueError("Result is outside the safe range")
            return result
        raise ValueError("Only numbers, parentheses, and arithmetic operators are allowed")

    result = evaluate(tree)
    return int(result) if result.is_integer() else result


@tool("calculator")
def calculator(expression: str) -> str:
    """Calculate exact arithmetic for quantities, portions, percentages, or unit conversions.

    Use operators +, -, *, /, //, %, ** and parentheses. Examples: `250 / 60`,
    `0.25 * 1000` (kg to g), or `(138 / 60) * 250` (calories for 250 g).
    Always use this tool instead of mental arithmetic when a numeric result affects
    a diary operation or nutrition answer.
    """
    result = evaluate_expression(expression)
    return json.dumps({"expression": expression, "result": result})
