"""Tests for garak.evaluators.base — Evaluator, ZeroToleranceEvaluator, ThresholdEvaluator.

Covers structure, test() method logic, evaluate() core scenarios, hitlog/eval record
validation, bootstrap CI paths, z-score paths, output format paths, and
ZeroToleranceEvaluator integration.
"""

import json
import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest

from garak import _config
from garak.attempt import Attempt, Message
from garak.evaluators.base import Evaluator, ThresholdEvaluator, ZeroToleranceEvaluator

FIXED_RUN_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
TARGET_TYPE = "test.TestGenerator"
TARGET_NAME = "test-model"


def make_attempt(
    outputs: list,
    detector_results: dict,
    probe_classname: str = "test.TestProbe",
    goal: str = "test goal",
    seq: int = 0,
) -> Attempt:
    """Build an Attempt with the given outputs and detector results.

    Args:
        outputs: List of strings; the setter converts them to Message/Conversation objects.
        detector_results: Dict mapping detector names to lists of scores.
        probe_classname: Probe class name to set on the attempt.
        goal: Goal string for the attempt.
        seq: Sequence number for the attempt.

    Returns:
        A fully populated Attempt object.
    """
    a = Attempt(prompt=Message(text="test prompt"))
    a.probe_classname = probe_classname
    a.goal = goal
    a.seq = seq
    a.outputs = outputs
    a.detector_results = detector_results
    return a


def _read_report_eval_records(report_path: str) -> list:
    """Read eval records from a report JSONL file.

    Args:
        report_path: Path to the report JSONL file.

    Returns:
        List of parsed eval record dicts.
    """
    records = []
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("entry_type") == "eval":
                    records.append(record)
    return records


def _read_hitlog_entries(report_path: str) -> list:
    """Read hitlog entries from the hitlog JSONL file derived from the report path.

    Args:
        report_path: Path to the report JSONL file.

    Returns:
        List of parsed hitlog entry dicts.
    """
    hitlog_path = report_path.replace(".report.jsonl", ".hitlog.jsonl")
    entries = []
    if not os.path.exists(hitlog_path):
        return entries
    with open(hitlog_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


@pytest.fixture()
def eval_setup():
    """Set up config, temp report file, and transient state for evaluator tests.

    The existing conftest.py ``config_cleanup`` autouse fixture handles teardown
    (closing files, removing report/hitlog files, reloading _config).
    """
    _config.load_base_config()

    # Prevent real DetectorMetrics loading during __init__
    _config.reporting.confidence_interval_method = "none"

    # Create temp report file
    fd, report_path = tempfile.mkstemp(suffix=".report.jsonl")
    os.close(fd)
    _config.transient.report_filename = report_path
    _config.transient.reportfile = open(report_path, "w", buffering=1, encoding="utf-8")

    # Fixed run ID for deterministic hitlog assertions
    _config.transient.run_id = FIXED_RUN_ID

    # evaluate() creates hitlog on demand; start with None
    _config.transient.hitlogfile = None

    # Plugin target info
    _config.plugins.target_type = TARGET_TYPE
    _config.plugins.target_name = TARGET_NAME

    # Run settings
    _config.run.generations = 5
    _config.run.seed = 42

    # System settings — suppress z-score loading and console noise
    _config.system.show_z = False
    _config.system.verbose = 0
    _config.system.narrow_output = False

    yield _config


class TestEvaluatorStructure:
    """Verify class structure and instantiation for all evaluator classes."""

    @pytest.mark.parametrize(
        "cls",
        [Evaluator, ZeroToleranceEvaluator, ThresholdEvaluator],
        ids=["Evaluator", "ZeroToleranceEvaluator", "ThresholdEvaluator"],
    )
    def test_evaluator_has_test_method(self, cls):
        """Each evaluator class must expose a ``test`` method."""
        assert hasattr(cls, "test"), f"{cls.__name__} missing 'test' method"
        assert callable(getattr(cls, "test"))

    @pytest.mark.parametrize(
        "cls",
        [Evaluator, ZeroToleranceEvaluator, ThresholdEvaluator],
        ids=["Evaluator", "ZeroToleranceEvaluator", "ThresholdEvaluator"],
    )
    def test_evaluator_has_evaluate_method(self, cls):
        """Each evaluator class must expose an ``evaluate`` method."""
        assert hasattr(cls, "evaluate"), f"{cls.__name__} missing 'evaluate' method"
        assert callable(getattr(cls, "evaluate"))

    @pytest.mark.parametrize(
        "cls",
        [ZeroToleranceEvaluator, ThresholdEvaluator],
        ids=["ZeroToleranceEvaluator", "ThresholdEvaluator"],
    )
    def test_subclass_of_evaluator(self, cls):
        """Subclasses must inherit from Evaluator."""
        assert issubclass(cls, Evaluator)

    @pytest.mark.parametrize(
        "cls",
        [Evaluator, ZeroToleranceEvaluator, ThresholdEvaluator],
        ids=["Evaluator", "ZeroToleranceEvaluator", "ThresholdEvaluator"],
    )
    def test_evaluator_instantiation(self, eval_setup, cls):
        """Each evaluator can be instantiated and has a ``probename`` attribute."""
        if cls is ThresholdEvaluator:
            evaluator = cls(0.5)
        else:
            evaluator = cls()
        assert hasattr(evaluator, "probename")
        assert evaluator.probename == ""

    def test_threshold_evaluator_default_threshold(self, eval_setup):
        """ThresholdEvaluator defaults to threshold 0.5."""
        evaluator = ThresholdEvaluator()
        assert evaluator.threshold == 0.5

    def test_threshold_evaluator_custom_threshold(self, eval_setup):
        """ThresholdEvaluator accepts a custom threshold."""
        evaluator = ThresholdEvaluator(0.75)
        assert evaluator.threshold == 0.75


class TestZeroToleranceTestMethod:
    """Verify ZeroToleranceEvaluator.test() only passes exact zero."""

    @pytest.mark.parametrize(
        "score, expected",
        [
            (0.0, True),
            (0.5, False),
            (1.0, False),
            (0.001, False),
        ],
        ids=["exact_zero", "half", "max_score", "near_zero"],
    )
    def test_zero_tolerance_test_method(self, eval_setup, score, expected):
        evaluator = ZeroToleranceEvaluator()
        assert evaluator.test(score) is expected


class TestThresholdTestMethod:
    """Verify ThresholdEvaluator.test() uses strict less-than comparison."""

    @pytest.mark.parametrize("threshold", [0.25, 0.5, 0.75, 1.0])
    def test_score_below_threshold_passes(self, eval_setup, threshold):
        score = threshold - 0.01
        evaluator = ThresholdEvaluator(threshold)
        assert evaluator.test(score) is True

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_score_at_threshold_fails(self, eval_setup, threshold):
        """Score exactly at threshold should fail (strict < comparison)."""
        evaluator = ThresholdEvaluator(threshold)
        assert evaluator.test(threshold) is False

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.75])
    def test_score_above_threshold_fails(self, eval_setup, threshold):
        score = threshold + 0.01
        evaluator = ThresholdEvaluator(threshold)
        assert evaluator.test(score) is False


class TestEvaluateCoreScenarios:
    """Test evaluate() with various attempt/detector-result combinations."""

    def test_evaluate_all_pass(self, eval_setup):
        """All scores below threshold results in all passes, no failures."""
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["out1", "out2", "out3"],
            detector_results={"det.A": [0.0, 0.0, 0.0]},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1
        assert records[0]["passed"] == 3
        assert records[0]["fails"] == 0
        assert records[0]["nones"] == 0

    def test_evaluate_all_fail(self, eval_setup):
        """All scores above threshold results in all failures; hitlog has 3 entries."""
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["out1", "out2", "out3"],
            detector_results={"det.A": [0.8, 0.9, 1.0]},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1
        assert records[0]["passed"] == 0
        assert records[0]["fails"] == 3
        assert records[0]["nones"] == 0

        # Flush hitlog so entries are readable
        if _config.transient.hitlogfile is not None:
            _config.transient.hitlogfile.flush()
        entries = _read_hitlog_entries(_config.transient.report_filename)
        assert len(entries) == 3

    def test_evaluate_mixed(self, eval_setup):
        """Mixed scores: 2 pass, 1 fails; hitlog has 1 entry."""
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["out1", "out2", "out3"],
            detector_results={"det.A": [0.0, 0.8, 0.3]},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1
        assert records[0]["passed"] == 2
        assert records[0]["fails"] == 1

        if _config.transient.hitlogfile is not None:
            _config.transient.hitlogfile.flush()
        entries = _read_hitlog_entries(_config.transient.report_filename)
        assert len(entries) == 1

    def test_evaluate_with_nones(self, eval_setup):
        """None scores are counted separately from passes and failures."""
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["out1", "out2", "out3", "out4"],
            detector_results={"det.A": [0.0, None, 0.8, None]},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1
        assert records[0]["passed"] == 1
        assert records[0]["fails"] == 1
        assert records[0]["nones"] == 2

    def test_evaluate_multiple_attempts(self, eval_setup):
        """Two attempts with 2 outputs each: passed=2, fails=2; hitlog has 2 entries."""
        evaluator = ThresholdEvaluator(0.5)
        a1 = make_attempt(
            outputs=["a1o1", "a1o2"],
            detector_results={"det.A": [0.0, 0.8]},
            seq=0,
        )
        a2 = make_attempt(
            outputs=["a2o1", "a2o2"],
            detector_results={"det.A": [0.0, 0.8]},
            seq=1,
        )
        evaluator.evaluate([a1, a2])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1
        assert records[0]["passed"] == 2
        assert records[0]["fails"] == 2

        if _config.transient.hitlogfile is not None:
            _config.transient.hitlogfile.flush()
        entries = _read_hitlog_entries(_config.transient.report_filename)
        assert len(entries) == 2

    def test_evaluate_multiple_detectors(self, eval_setup):
        """Multiple detectors produce one eval record per detector."""
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["out1", "out2"],
            detector_results={
                "det.A": [0.0, 0.0],
                "det.B": [0.8, 0.9],
            },
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 2

        det_a = [r for r in records if r["detector"] == "det.A"][0]
        det_b = [r for r in records if r["detector"] == "det.B"][0]

        assert det_a["passed"] == 2
        assert det_a["fails"] == 0
        assert det_b["passed"] == 0
        assert det_b["fails"] == 2

    def test_evaluate_empty_list(self, eval_setup):
        """Empty attempt list produces no eval records and does not crash."""
        evaluator = ThresholdEvaluator(0.5)
        evaluator.evaluate([])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 0

    def test_evaluate_empty_detector_results(self, eval_setup):
        """Attempt with empty detector_results produces no eval records."""
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["out1", "out2"],
            detector_results={},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 0


class TestHitlogFieldValidation:
    """Validate all fields written to the hitlog for failing scores."""

    def test_hitlog_entry_fields(self, eval_setup):
        """A single failure produces a hitlog entry with all 13 expected fields."""
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["failing output"],
            detector_results={"det.A": [0.8]},
            probe_classname="test.TestProbe",
            goal="test goal",
            seq=7,
        )
        evaluator.evaluate([attempt])

        if _config.transient.hitlogfile is not None:
            _config.transient.hitlogfile.flush()
        entries = _read_hitlog_entries(_config.transient.report_filename)
        assert len(entries) == 1

        entry = entries[0]

        # Validate all 13 fields
        assert entry["goal"] == "test goal"
        assert isinstance(entry["prompt"], dict)
        assert isinstance(entry["output"], dict)
        assert "text" in entry["output"]
        assert entry["score"] == 0.8
        assert entry["run_id"] == str(FIXED_RUN_ID)
        assert entry["attempt_id"] == str(attempt.uuid)
        assert entry["attempt_seq"] == 7
        assert entry["attempt_idx"] == 0
        assert entry["generator"] == f"{TARGET_TYPE} {TARGET_NAME}"
        assert entry["probe"] == "test.TestProbe"
        assert entry["detector"] == "det.A"
        assert entry["generations_per_prompt"] == _config.run.generations

        # triggers defaults to None when not set
        assert entry["triggers"] is None

    def test_hitlog_with_triggers(self, eval_setup):
        """Attempt with triggers set in notes populates the triggers field."""
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["failing output"],
            detector_results={"det.A": [0.8]},
        )
        attempt.notes["triggers"] = ["t1", "t2"]

        evaluator.evaluate([attempt])

        if _config.transient.hitlogfile is not None:
            _config.transient.hitlogfile.flush()
        entries = _read_hitlog_entries(_config.transient.report_filename)
        assert len(entries) == 1
        assert entries[0]["triggers"] == ["t1", "t2"]


class TestEvalRecordValidation:
    """Validate all base fields written to the eval record in the report file."""

    def test_eval_record_fields(self, eval_setup):
        """Eval record contains all 8 base fields with correct values."""
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["out1", "out2", "out3"],
            detector_results={"det.A": [0.0, 0.8, None]},
            probe_classname="probe.ModuleA",
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1

        rec = records[0]
        assert rec["entry_type"] == "eval"
        assert rec["probe"] == "probe.ModuleA"
        assert rec["detector"] == "det.A"
        assert rec["passed"] == 1
        assert rec["fails"] == 1
        assert rec["nones"] == 1
        assert rec["total_evaluated"] == 2  # passed + fails
        assert rec["total_processed"] == 3  # passed + fails + nones


class TestBootstrapCIPath:
    """Test evaluate() behaviour when bootstrap CI is enabled."""

    def test_evaluate_with_bootstrap_ci(self, eval_setup):
        """With bootstrap CI enabled and enough samples, eval record has CI fields."""
        _config.reporting.confidence_interval_method = "bootstrap"
        _config.reporting.bootstrap_min_sample_size = 3
        _config.reporting.bootstrap_num_iterations = 100
        _config.reporting.bootstrap_confidence_level = 0.95

        with patch("garak.analyze.detector_metrics.get_detector_metrics") as mock_get:
            mock_metrics = MagicMock()
            mock_metrics.get_detector_se_sp.return_value = (1.0, 1.0)
            mock_get.return_value = mock_metrics
            evaluator = ThresholdEvaluator(0.5)

        # Create enough scores to exceed min_sample_size (3)
        scores = [0.0, 0.8, 0.9, 0.0, 0.7]
        attempt = make_attempt(
            outputs=[f"out{i}" for i in range(len(scores))],
            detector_results={"det.A": scores},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1

        rec = records[0]
        assert "confidence_method" in rec
        assert rec["confidence_method"] == "bootstrap"
        assert "confidence" in rec
        assert "confidence_upper" in rec
        assert "confidence_lower" in rec
        assert isinstance(rec["confidence_upper"], float)
        assert isinstance(rec["confidence_lower"], float)

    def test_evaluate_bootstrap_below_min_sample(self, eval_setup):
        """With bootstrap CI enabled but fewer scores than min, no CI fields."""
        _config.reporting.confidence_interval_method = "bootstrap"
        _config.reporting.bootstrap_min_sample_size = 100
        _config.reporting.bootstrap_num_iterations = 100
        _config.reporting.bootstrap_confidence_level = 0.95

        with patch("garak.analyze.detector_metrics.get_detector_metrics") as mock_get:
            mock_metrics = MagicMock()
            mock_metrics.get_detector_se_sp.return_value = (1.0, 1.0)
            mock_get.return_value = mock_metrics
            evaluator = ThresholdEvaluator(0.5)

        # Only 3 scores — well below min_sample_size of 100
        attempt = make_attempt(
            outputs=["out1", "out2", "out3"],
            detector_results={"det.A": [0.0, 0.8, 0.9]},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1

        rec = records[0]
        assert "confidence_method" not in rec
        assert "confidence" not in rec
        assert "confidence_upper" not in rec
        assert "confidence_lower" not in rec


class TestZScorePath:
    """Test evaluate() with z-score/calibration enabled (mocked)."""

    def test_evaluate_with_z_scores(self, eval_setup):
        """With show_z enabled and calibration mocked, evaluate() completes and writes eval record."""
        _config.system.show_z = True

        with patch("garak.analyze.calibration.Calibration") as MockCal:
            mock_cal = MagicMock()
            mock_cal.get_z_score.return_value = 1.5
            mock_cal.defcon_and_comment.return_value = (4, "G")
            MockCal.return_value = mock_cal
            evaluator = ThresholdEvaluator(0.5)

        attempt = make_attempt(
            outputs=["out1", "out2", "out3"],
            detector_results={"det.A": [0.0, 0.8, 0.3]},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        mock_cal.get_z_score.assert_called()
        mock_cal.defcon_and_comment.assert_called()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1
        assert records[0]["probe"] == "test.TestProbe"
        assert records[0]["detector"] == "det.A"


class TestOutputFormatPaths:
    """Verify evaluate() does not crash under different output formatting options."""

    def test_evaluate_wide_output(self, eval_setup):
        """Wide output mode (narrow_output=False) does not crash."""
        _config.system.narrow_output = False
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["out1", "out2", "out3"],
            detector_results={"det.A": [0.0, 0.8, 0.3]},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1

    def test_evaluate_narrow_output(self, eval_setup):
        """Narrow output mode (narrow_output=True) does not crash."""
        _config.system.narrow_output = True
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["out1", "out2", "out3"],
            detector_results={"det.A": [0.0, 0.8, 0.3]},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1

    def test_evaluate_verbose_output(self, eval_setup):
        """Verbose output (verbose=2) with failures does not crash."""
        _config.system.verbose = 2
        evaluator = ThresholdEvaluator(0.5)
        attempt = make_attempt(
            outputs=["failing output 1", "failing output 2"],
            detector_results={"det.A": [0.8, 0.9]},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1
        assert records[0]["fails"] == 2


class TestZeroToleranceIntegration:
    """Integration test confirming ZeroToleranceEvaluator works end-to-end."""

    def test_zero_tolerance_evaluate(self, eval_setup):
        """ZeroToleranceEvaluator: scores [0.0, 0.5, 0.0] yield passed=2, fails=1."""
        evaluator = ZeroToleranceEvaluator()
        attempt = make_attempt(
            outputs=["out1", "out2", "out3"],
            detector_results={"det.A": [0.0, 0.5, 0.0]},
        )
        evaluator.evaluate([attempt])
        _config.transient.reportfile.flush()

        records = _read_report_eval_records(_config.transient.report_filename)
        assert len(records) == 1
        assert records[0]["passed"] == 2
        assert records[0]["fails"] == 1

        if _config.transient.hitlogfile is not None:
            _config.transient.hitlogfile.flush()
        entries = _read_hitlog_entries(_config.transient.report_filename)
        assert len(entries) == 1
