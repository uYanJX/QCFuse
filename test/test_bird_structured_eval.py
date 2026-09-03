"""CPU-only tests for the BIRD execution-accuracy evaluator."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from blend.structured_eval import BirdExecutionEvaluator


class TestBirdExecutionEvaluator(unittest.TestCase):
    def test_execution_accuracy_uses_read_only_result_set_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_dir = root / "databases" / "demo"
            database_dir.mkdir(parents=True)
            database_path = database_dir / "demo.sqlite"
            with sqlite3.connect(database_path) as connection:
                connection.execute("CREATE TABLE items (id INTEGER, name TEXT)")
                connection.executemany(
                    "INSERT INTO items VALUES (?, ?)",
                    [(1, "alpha"), (2, "beta")],
                )

            gold_sql = "SELECT name FROM items"
            source_path = root / "mini_dev_sqlite.json"
            source_path.write_text(
                json.dumps(
                    [
                        {
                            "db_id": "demo",
                            "question": "List item names.",
                            "SQL": gold_sql,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            dataset = [
                {
                    "input": "List item names.",
                    "context": ["Table: items"],
                    "answers": [gold_sql],
                    "num_chunks": 1,
                }
            ]
            evaluator = BirdExecutionEvaluator(
                dataset,
                source_path,
                root / "databases",
                expected_rows=1,
            )

            self.assertEqual(
                evaluator.score(
                    0,
                    "SELECT name FROM items ORDER BY name DESC",
                    [gold_sql],
                ),
                1.0,
            )
            self.assertEqual(
                evaluator.score(
                    0,
                    "Here is the query:\n```sql\nSELECT name FROM items\n```",
                    [gold_sql],
                ),
                1.0,
            )
            self.assertEqual(
                evaluator.score(0, "SELECT id FROM items", [gold_sql]),
                0.0,
            )
            self.assertEqual(
                evaluator.score(0, "DROP TABLE items", [gold_sql]),
                0.0,
            )
            with sqlite3.connect(database_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
