# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import garak._plugins
import garak.probes.base

DONOTANSWER_PROBES = [
    classname
    for (classname, active) in garak._plugins.enumerate_plugins("probes")
    if classname.startswith("probes.donotanswer")
]


@pytest.mark.parametrize("classname", DONOTANSWER_PROBES)
def test_donotanswer_load(classname):
    p = garak._plugins.load_plugin(classname)
    assert isinstance(p, garak.probes.base.Probe)


@pytest.mark.parametrize("classname", DONOTANSWER_PROBES)
def test_donotanswer_prompts_not_empty(classname):
    p = garak._plugins.load_plugin(classname)
    assert len(p.prompts) > 0, "Probe must have more than zero prompts"
