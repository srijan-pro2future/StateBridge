"""Regression tests for publication-critical evaluation harness behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


STATEBRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STATEBRIDGE_ROOT))

from methods.state_bridge import StateBridge, select_dataset_rows  # noqa: E402
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

    def test_exact_item_selection_preserves_dataset_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.json"
            path.write_text(json.dumps({"item_ids": [3, 1]}), encoding="utf-8")
            args = argparse.Namespace(indices_file=str(path), limit=None)
            dataset = [{"question": str(index)} for index in range(5)]
            with patch("methods.state_bridge.load_dataset_by_name", return_value=dataset):
                selected = select_dataset_rows(args, "gpqa")

            self.assertEqual([row["_dataset_idx"] for row in selected], [3, 1])
            self.assertEqual(args._selected_item_ids, [3, 1])
            self.assertEqual(args._indices_sha256, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
