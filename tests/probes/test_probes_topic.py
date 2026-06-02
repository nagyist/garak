# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
import wn

import garak._plugins
import garak.probes.base
import garak.probes.topic

TEST_LEXICON = "oewn:2023"
TEST_TERM = "abortion"
TEST_SYNSET_ID = "oewn-00231191-n"


@pytest.fixture(scope="module")
def sysnet():
    garak.probes.topic.WordnetBlockedWords()
    wn.download(TEST_LEXICON)
    w = wn.Wordnet(TEST_LEXICON)
    s = w.synset(TEST_SYNSET_ID)
    return s


PROBES = [
    classname
    for (classname, active) in garak._plugins.enumerate_plugins("probes")
    if ".topic.Wordnet" in classname
]


@pytest.mark.parametrize("probename", PROBES)
def test_topic_wordnet_load(probename):
    p = garak._plugins.load_plugin(probename)
    assert isinstance(p, garak.probes.base.Probe)


@pytest.mark.parametrize("probename", PROBES)
def test_topic_wordnet_version(probename):
    p = garak._plugins.load_plugin(probename)
    assert p.lexicon == TEST_LEXICON


@pytest.mark.parametrize("probename", PROBES)
def test_topic_wordnet_get_node_terms(probename, sysnet):
    p = garak._plugins.load_plugin(probename)
    terms = p._get_node_terms(sysnet)
    assert list(terms) == ["abortion"]


@pytest.mark.parametrize("probename", PROBES)
def test_topic_wordnet_get_node_children(probename, sysnet):
    p = garak._plugins.load_plugin(probename)
    children = p._get_node_children(sysnet)
    assert children == [wn.synset("oewn-00231342-n"), wn.synset("oewn-00232028-n")]


@pytest.mark.parametrize("probename", PROBES)
def test_topic_wordnet_get_node_id(probename, sysnet):
    p = garak._plugins.load_plugin(probename)
    assert p._get_node_id(sysnet) == TEST_SYNSET_ID


def test_topic_wordnet_blocklist_get_initial_nodes(sysnet):
    p = garak.probes.topic.WordnetBlockedWords()
    p.target_topic = TEST_TERM
    initial_nodes = p._get_initial_nodes()
    assert initial_nodes == [
        sysnet,
        wn.synset("oewn-07334252-n"),
        wn.synset("oewn-00231342-n"),
        wn.synset("oewn-00232028-n"),
    ]


@pytest.mark.parametrize(
    "first_call_error",
    [
        sqlite3.OperationalError("no such table: lexicons"),
        wn.Error("no lexicon found with lang=None and lexicon='oewn:2023'"),
    ],
    ids=["sqlite_missing_database", "wn_missing_lexicon"],
)
def test_topic_wordnet_downloads_missing_lexicon(first_call_error):
    # when the lexicon cannot be opened the probe should download it and retry,
    # rather than letting the underlying error propagate and fail to load.
    # wn.Error covers the case where the database exists but the requested
    # lexicon is not installed, see https://github.com/NVIDIA/garak/issues/1230
    loaded_wordnet = MagicMock(name="wn.Wordnet")
    download_path = MagicMock(name="download_tempfile_path")

    with (
        patch.object(
            wn, "Wordnet", side_effect=[first_call_error, loaded_wordnet]
        ) as mock_wordnet,
        patch.object(wn, "download", return_value=download_path) as mock_download,
        patch.object(pathlib.Path, "rmdir"),
    ):
        probe = garak.probes.topic.WordnetBlockedWords()

    assert (
        mock_download.call_count == 1
    ), "a missing lexicon should trigger exactly one download"
    assert (
        mock_wordnet.call_count == 2
    ), "the lexicon should be reloaded after downloading it"
    assert (
        probe.w is loaded_wordnet
    ), "the probe should keep the reloaded wordnet after recovery"
    download_path.unlink.assert_called_once()
