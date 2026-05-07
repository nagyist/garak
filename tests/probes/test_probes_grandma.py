# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import garak._plugins
import garak.payloads
import garak.probes.grandma

GRANDMA_PROBES = (
    "probes.grandma.Win10",
    "probes.grandma.Win11",
    "probes.grandma.Slurs",
    "probes.grandma.Substances",
)


def test_grandma_win10_prompts_contain_product_names():
    p = garak._plugins.load_plugin("probes.grandma.Win10")
    product_names = garak.payloads.load(p.payload).payloads
    for product_name in product_names:
        matching = [prompt for prompt in p.prompts if product_name in prompt]
        assert len(matching) > 0, f"Expected prompts containing '{product_name}'"


def test_grandma_win11_inherits_win10():
    assert issubclass(garak.probes.grandma.Win11, garak.probes.grandma.Win10)


def test_grandma_slurs_prompts_contain_groups():
    p = garak._plugins.load_plugin("probes.grandma.Slurs")
    for group in p.human_groups:
        matching = [prompt for prompt in p.prompts if group in prompt]
        assert len(matching) > 0, f"Expected prompts containing '{group}'"


def test_grandma_substances_prompts_contain_substances():
    p = garak._plugins.load_plugin("probes.grandma.Substances")
    for substance in p.substances:
        matching = [prompt for prompt in p.prompts if substance in prompt]
        assert len(matching) > 0, f"Expected prompts containing '{substance}'"
