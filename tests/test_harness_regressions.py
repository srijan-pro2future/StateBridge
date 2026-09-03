"""Regression tests for publication-critical evaluation harness behavior."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


STATEBRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STATEBRIDGE_ROOT))

from methods.state_bridge import StateBridge  # noqa: E402
from utils import set_seed  # noqa: E402


class HarnessRegressionTests(unittest.TestCase):
    def test_large_derived_seed_is_normalized(self) -> None:
        set_seed(9001 * 1_000_003)
        self.assertEqual(os.environ["PYTHONHASHSEED"], str((9001 * 1_000_003) % (2**32)))

    def test_gpqa_rejects_numeric_tail_as_answer(self) -> None:
        evaluator = object.__new__(StateBridge)
        evaluator.task = "gpqa"
        self.assertIsNone(evaluator._extract_answer("The result is 21.", {}))

    def test_gpqa_accepts_boxed_choice(self) -> None:
        evaluator = object.__new__(StateBridge)
        evaluator.task = "gpqa"
        self.assertEqual(evaluator._extract_answer(r"Answer: \boxed{D}", {}), "d")


if __name__ == "__main__":
    unittest.main()
