# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import garak._plugins
import garak.probes.base

PHRASING_PROBES = (
    "probes.phrasing.PastTenseFull",
    "probes.phrasing.PastTense",
    "probes.phrasing.FutureTenseFull",
    "probes.phrasing.FutureTense",
)


@pytest.mark.parametrize("classname", PHRASING_PROBES)
def test_phrasing_load(classname):
    p = garak._plugins.load_plugin(classname)
    assert isinstance(p, garak.probes.base.Probe)


@pytest.mark.parametrize("classname", PHRASING_PROBES)
def test_phrasing_prompts_not_empty(classname):
    p = garak._plugins.load_plugin(classname)
    assert len(p.prompts) > 0, "Probe must have more than zero prompts"


def test_phrasing_pruned_probes_have_fewer_prompts():
    p_full = garak._plugins.load_plugin("probes.phrasing.PastTenseFull")
    p_pruned = garak._plugins.load_plugin("probes.phrasing.PastTense")
    assert len(p_pruned.prompts) <= len(p_full.prompts)


def test_phrasing_pruned_prompt_count_within_cap():
    p = garak._plugins.load_plugin("probes.phrasing.PastTense")
    assert len(p.prompts) <= p.soft_probe_prompt_cap
