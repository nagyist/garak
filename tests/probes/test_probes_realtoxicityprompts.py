# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import garak._plugins
import garak.probes.base
import garak.probes.realtoxicityprompts

RTP_DYNAMIC_PROBES = [
    classname
    for (classname, active) in garak._plugins.enumerate_plugins("probes")
    if ".realtoxicityprompts.RTP" in classname
    and classname != "probes.realtoxicityprompts.RTPBlank"
]


def test_rtp_blank_load():
    p = garak._plugins.load_plugin("probes.realtoxicityprompts.RTPBlank")
    assert isinstance(p, garak.probes.base.Probe)


def test_rtp_blank_prompts():
    p = garak._plugins.load_plugin("probes.realtoxicityprompts.RTPBlank")
    assert len(p.prompts) == 5
    assert p.prompts[1:] == [".", "<Links>", "<Wiki>", ""]


@pytest.mark.parametrize("classname", RTP_DYNAMIC_PROBES)
def test_rtp_dynamic_load(classname):
    p = garak._plugins.load_plugin(classname)
    assert isinstance(p, garak.probes.base.Probe)


@pytest.mark.parametrize("classname", RTP_DYNAMIC_PROBES)
def test_rtp_dynamic_prompts_not_empty(classname):
    p = garak._plugins.load_plugin(classname)
    assert len(p.prompts) > 0, "Probe must have more than zero prompts"
