import re

import pytest

from garak import cli
from garak.generators.base import Generator
from garak.generators.function import Multiple, Single


def passed_function(prompt: str, **kwargs):
    return [""]


def passed_function_multi(prompt: str, generations: int, **kwargs):
    return [""] * generations


def test_function_single(capsys):

    args = [
        "-m",
        "function",
        "-n",
        f"{__name__}#passed_function",
        "-p",
        "test.Blank",
    ]
    cli.main(args)
    result = capsys.readouterr()
    last_line = result.out.strip().split("\n")[-1]
    assert re.match("^✔️  garak run complete in [0-9]+\\.[0-9]+s$", last_line)


@pytest.mark.parametrize(
    ("klass", "name"),
    [
        (Single, f"{__name__}#passed_function"),
        (Multiple, f"{__name__}#passed_function_multi"),
    ],
)
def test_function_inherits_base_default_params(klass, name):
    # regression for #1096: the function generators overrode DEFAULT_PARAMS
    # with only `kwargs`, so base params such as `max_tokens` were never set
    # as instance attributes and access raised AttributeError.
    g = klass(name=name)
    for param in Generator.DEFAULT_PARAMS:
        assert hasattr(
            g, param
        ), f"{klass.__name__} is missing inherited param '{param}'"
    assert g.max_tokens == Generator.DEFAULT_PARAMS["max_tokens"]
    # the function-specific default must still be present
    assert g.kwargs == {}


def test_function_multiple(capsys):

    args = [
        "-m",
        "function.Multiple",
        "-n",
        f"{__name__}#passed_function_multi",
        "-p",
        "test.Blank",
    ]
    cli.main(args)
    result = capsys.readouterr()
    last_line = result.out.strip().split("\n")[-1]
    assert re.match("^✔️  garak run complete in [0-9]+\\.[0-9]+s$", last_line)
