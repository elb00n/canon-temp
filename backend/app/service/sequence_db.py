"""Compatibility wrapper for sequence DB helpers.

The actual implementation now lives in db.database so the backend can use a
single SQLite file for both inspection logs and sequence run storage.
"""

from db.database import DEFAULT_DB_PATH, StoredSequenceRun, connect, initialize_sequence_db as initialize, upsert_sequence_run

__all__ = [
	"DEFAULT_DB_PATH",
	"StoredSequenceRun",
	"connect",
	"initialize",
	"upsert_sequence_run",
]