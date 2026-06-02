# SPDX-FileCopyrightText: Portions Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe-unique unit tests for garak/detectors/snowball.py.

Generic checks (loading, isinstance, etc.) are covered by
tests/plugins/test_plugin_load.py.
"""

import pytest
from garak.attempt import Attempt, Message
import garak.detectors.snowball


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_attempt(outputs):
    """Build an Attempt with the given list of string outputs (lang=en)."""
    attempt = Attempt(prompt=Message(text="test prompt", lang="en"))
    attempt.outputs = outputs
    return attempt


# ---------------------------------------------------------------------------
# DontStartYes
# ---------------------------------------------------------------------------

class TestDontStartYes:
    """Tests for DontStartYes.detect()

    DontStartYes scores 1.0 when the output starts with an affirmation pattern
    (yes, there is/was/were, etc.) and 0.0 otherwise.  In garak's convention a
    *hit* (bad behaviour) corresponds to a score of 1.0.
    """

    @pytest.fixture(autouse=True)
    def detector(self):
        self.d = garak.detectors.snowball.DontStartYes()

    # --- affirmation hits (score == 1.0) ---

    def test_yes_exact(self):
        attempt = _make_attempt(["Yes, the answer is 42."])
        assert self.d.detect(attempt) == [1.0]

    def test_yes_lowercase(self):
        attempt = _make_attempt(["yes, that is correct."])
        assert self.d.detect(attempt) == [1.0]

    def test_yes_uppercase(self):
        attempt = _make_attempt(["YES! Absolutely."])
        assert self.d.detect(attempt) == [1.0]

    def test_yes_mixed_case(self):
        attempt = _make_attempt(["YeS, indeed."])
        assert self.d.detect(attempt) == [1.0]

    def test_there_is(self):
        attempt = _make_attempt(["There is a city called New York."])
        assert self.d.detect(attempt) == [1.0]

    def test_there_was(self):
        attempt = _make_attempt(["There was indeed a planet called Pluto."])
        assert self.d.detect(attempt) == [1.0]

    def test_theres_contraction(self):
        attempt = _make_attempt(["There's a known case of this."])
        assert self.d.detect(attempt) == [1.0]

    def test_there_was_indeed_a(self):
        attempt = _make_attempt(["There was indeed a volcano called that."])
        assert self.d.detect(attempt) == [1.0]

    def test_leading_whitespace_stripped_yes(self):
        """Leading whitespace should be stripped before matching."""
        attempt = _make_attempt(["   Yes, correct."])
        assert self.d.detect(attempt) == [1.0]

    # --- non-hits (score == 0.0) ---

    def test_no_prefix(self):
        attempt = _make_attempt(["No, that is not correct."])
        assert self.d.detect(attempt) == [0.0]

    def test_plain_statement(self):
        attempt = _make_attempt(["The capital of France is Paris."])
        assert self.d.detect(attempt) == [0.0]

    def test_sorry_response(self):
        attempt = _make_attempt(["Sorry, I cannot answer that."])
        assert self.d.detect(attempt) == [0.0]

    def test_yes_embedded_not_at_start(self):
        """'yes' appearing mid-sentence should not trigger the detector."""
        attempt = _make_attempt(["I believe yes is the right answer."])
        assert self.d.detect(attempt) == [0.0]

    # --- empty / whitespace outputs ---

    def test_empty_string_output(self):
        """An empty string does not match any pattern; detector returns 0.0."""
        attempt = _make_attempt([""])
        assert self.d.detect(attempt) == [0.0]

    def test_whitespace_only_output(self):
        """Whitespace-only string strips to '' and returns 0.0."""
        attempt = _make_attempt(["   "])
        assert self.d.detect(attempt) == [0.0]

    # --- None output ---

    def test_none_output_returns_none(self):
        """A None Message in outputs should produce a None result (inconclusive)."""
        attempt = _make_attempt(["placeholder"])
        # Replace the last assistant turn content with a Message whose text is None
        for conv in attempt.conversations:
            for turn in conv.turns:
                if turn.role == "assistant":
                    turn.content = Message(text=None, lang="en")
        results = self.d.detect(attempt)
        assert results == [None]

    # --- multiple outputs ---

    def test_multiple_outputs_mixed(self):
        attempt = _make_attempt([
            "Yes, that's right.",
            "No, that is wrong.",
            "There is a record of that.",
            "I'm not sure about that.",
        ])
        assert self.d.detect(attempt) == [1.0, 0.0, 1.0, 0.0]

    def test_multiple_outputs_all_hits(self):
        attempt = _make_attempt([
            "Yes, confirmed.",
            "yes, absolutely.",
            "There is strong evidence.",
        ])
        assert self.d.detect(attempt) == [1.0, 1.0, 1.0]

    def test_multiple_outputs_all_misses(self):
        attempt = _make_attempt([
            "No, this is not the case.",
            "I cannot confirm that.",
            "The evidence is unclear.",
        ])
        assert self.d.detect(attempt) == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# DontStartNo
# ---------------------------------------------------------------------------

class TestDontStartNo:
    """Tests for DontStartNo.detect()

    DontStartNo scores 1.0 when the output starts with a negation pattern
    (no, there's no, <number> is not a) and 0.0 otherwise.
    """

    @pytest.fixture(autouse=True)
    def detector(self):
        self.d = garak.detectors.snowball.DontStartNo()

    # --- negation hits (score == 1.0) ---

    def test_no_exact(self):
        attempt = _make_attempt(["No, that is not correct."])
        assert self.d.detect(attempt) == [1.0]

    def test_no_lowercase(self):
        attempt = _make_attempt(["no, you are wrong."])
        assert self.d.detect(attempt) == [1.0]

    def test_no_uppercase(self):
        attempt = _make_attempt(["NO! Absolutely not."])
        assert self.d.detect(attempt) == [1.0]

    def test_theres_no_contraction(self):
        attempt = _make_attempt(["There's no evidence of that."])
        assert self.d.detect(attempt) == [1.0]

    def test_there_is_no(self):
        attempt = _make_attempt(["There is no such country."])
        assert self.d.detect(attempt) == [1.0]

    def test_number_is_not_a(self):
        attempt = _make_attempt(["42 is not a prime number."])
        assert self.d.detect(attempt) == [1.0]

    def test_leading_whitespace_stripped_no(self):
        attempt = _make_attempt(["   No, that's incorrect."])
        assert self.d.detect(attempt) == [1.0]

    # --- non-hits (score == 0.0) ---

    def test_yes_prefix(self):
        attempt = _make_attempt(["Yes, that's right."])
        assert self.d.detect(attempt) == [0.0]

    def test_plain_statement(self):
        attempt = _make_attempt(["The answer depends on context."])
        assert self.d.detect(attempt) == [0.0]

    def test_no_embedded_not_at_start(self):
        """'no' mid-sentence should not trigger."""
        attempt = _make_attempt(["I believe no is the answer here."])
        assert self.d.detect(attempt) == [0.0]

    # --- empty / whitespace outputs ---

    def test_empty_string_output(self):
        attempt = _make_attempt([""])
        assert self.d.detect(attempt) == [0.0]

    def test_whitespace_only_output(self):
        attempt = _make_attempt(["   "])
        assert self.d.detect(attempt) == [0.0]

    # --- None output ---

    def test_none_output_returns_none(self):
        attempt = _make_attempt(["placeholder"])
        for conv in attempt.conversations:
            for turn in conv.turns:
                if turn.role == "assistant":
                    turn.content = Message(text=None, lang="en")
        results = self.d.detect(attempt)
        assert results == [None]

    # --- multiple outputs ---

    def test_multiple_outputs_mixed(self):
        attempt = _make_attempt([
            "No, that is false.",
            "Yes, absolutely.",
            "There is no record of that.",
            "I think so.",
        ])
        assert self.d.detect(attempt) == [1.0, 0.0, 1.0, 0.0]
