import json

import pytest

from mfp_agent.calculator import calculator, evaluate_expression


def test_quantity_calculation():
    assert evaluate_expression("250 / 60") == pytest.approx(4.1666666667)


def test_unit_conversion():
    assert evaluate_expression("0.25 * 1000") == 250


def test_tool_returns_structured_json():
    payload = json.loads(calculator.invoke({"expression": "(138 / 60) * 250"}))
    assert payload == {"expression": "(138 / 60) * 250", "result": 575}


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "open('/tmp/file')", "1 / 0", "2 ** 100"],
)
def test_unsafe_or_invalid_expressions_are_rejected(expression):
    with pytest.raises(ValueError):
        evaluate_expression(expression)
