import sys
import unittest
from unittest.mock import patch

from stochastic_world.cli import parse_args


class CliWorkersAliasTest(unittest.TestCase):
    def test_workers_alias_sets_hybrid_workers(self):
        with patch.object(sys, "argv", ["world.py", "--workers", "10"]):
            args = parse_args()
        self.assertEqual(args.hybrid_workers, 10)

    def test_hybrid_workers_name_still_works(self):
        with patch.object(sys, "argv", ["world.py", "--hybrid-workers", "7"]):
            args = parse_args()
        self.assertEqual(args.hybrid_workers, 7)


if __name__ == "__main__":
    unittest.main()
