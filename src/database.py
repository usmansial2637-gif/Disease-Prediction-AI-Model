"""
database.py
------------
SQLite storage for prediction history (Level 5).

Every prediction made through the Streamlit app gets logged here so users
(or a dashboard, Level 6) can look back at what was predicted, when, and
with how much confidence.

Table: predictions
    id                 INTEGER PRIMARY KEY AUTOINCREMENT
    dataset             TEXT     -- e.g. "heart"
    model_name          TEXT     -- e.g. "CatBoost"
    patient_age         REAL     -- NULL if the dataset has no age-like column
    symptoms            TEXT     -- JSON blob of every input feature -> value
    predicted_disease   TEXT     -- human-readable disease/outcome label
    prediction          INTEGER  -- raw 0/1 model output
    confidence          REAL     -- predicted probability of the positive class
    created_at          TEXT     -- ISO timestamp
"""

import json
import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "database", "history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    model_name TEXT NOT NULL,
    username TEXT,
    patient_age REAL,
    symptoms TEXT NOT NULL,
    predicted_disease TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL
);
"""


def _ensure_username_column(conn):
    """Migration for DBs created before Level 9 (auth) added this column."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    if "username" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN username TEXT")
        conn.commit()


def init_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    _ensure_username_column(conn)
    conn.close()


def _extract_age(feature_values: dict):
    """Looks for an age-like column (case-insensitive) among the inputs."""
    for k, v in feature_values.items():
        if k.lower() == "age":
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def log_prediction(dataset, model_name, feature_values: dict, predicted_disease,
                    prediction: int, confidence, username=None, db_path: str = DB_PATH):
    """Inserts one prediction record. Returns the new row's id."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO predictions
           (dataset, model_name, username, patient_age, symptoms, predicted_disease,
            prediction, confidence, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            dataset, model_name, username, _extract_age(feature_values),
            json.dumps(feature_values), predicted_disease, int(prediction),
            float(confidence) if confidence is not None else None,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def fetch_history(dataset=None, username=None, limit=200, db_path: str = DB_PATH):
    """Returns prediction history as a list of dicts, most recent first.
    Pass username to scope results to one user (e.g. for patient role)."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    conditions, params = [], []
    if dataset:
        conditions.append("dataset = ?")
        params.append(dataset)
    if username:
        conditions.append("username = ?")
        params.append(username)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    cur.execute(f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT ?", params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def clear_history(dataset=None, db_path: str = DB_PATH):
    """Deletes prediction history (optionally scoped to one dataset)."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    if dataset:
        conn.execute("DELETE FROM predictions WHERE dataset = ?", (dataset,))
    else:
        conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
