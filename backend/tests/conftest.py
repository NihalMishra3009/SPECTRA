"""Shared pytest configuration — keep the repo database/spectra.db pristine.

The SQLite runtime database is redirected to a throwaway temp dir for the
entire test session (individual tests may override via monkeypatch).
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("SPECTRA_DB_DIR", str(Path(tempfile.mkdtemp(prefix="spectra_test_db_"))))
os.environ.setdefault("SPECTRA_DB_FILE", str(Path(os.environ["SPECTRA_DB_DIR"]) / "test.db"))