"""
auth.py
-------
Simple username/password authentication with three roles: admin, doctor,
patient. Passwords are salted + hashed (PBKDF2-HMAC-SHA256) — never stored
in plain text.

A default admin account is seeded on first run:
    username: admin
    password: admin123
Change this immediately in a real deployment (see README).

Role permissions (enforced in app.py, not here — this module just answers
"who is this / what role do they have"):
    admin   - manage users, see all prediction history, full dashboard
    doctor  - see all prediction history, full dashboard, make predictions
    patient - make predictions, see only their own prediction history
"""

import hashlib
import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "database", "users.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'doctor', 'patient')),
    created_at TEXT NOT NULL
);
"""

ROLES = ["admin", "doctor", "patient"]


def _hash_password(password: str, salt: bytes = None):
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return salt.hex(), digest.hex()


def init_auth_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()

    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        salt_hex, hash_hex = _hash_password("admin123")
        conn.execute(
            "INSERT INTO users (username, salt, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("admin", salt_hex, hash_hex, "admin", datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
    conn.close()


def create_user(username: str, password: str, role: str, db_path: str = DB_PATH):
    if role not in ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {ROLES}.")
    init_auth_db(db_path)
    salt_hex, hash_hex = _hash_password(password)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO users (username, salt, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, salt_hex, hash_hex, role, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        return True, "User created."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists."
    finally:
        conn.close()


def verify_login(username: str, password: str, db_path: str = DB_PATH):
    """Returns the user's role (str) if credentials are correct, else None."""
    init_auth_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    salt = bytes.fromhex(row["salt"])
    _, hash_hex = _hash_password(password, salt)
    if hash_hex == row["password_hash"]:
        return row["role"]
    return None


def list_users(db_path: str = DB_PATH):
    init_auth_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT id, username, role, created_at FROM users ORDER BY id")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def delete_user(username: str, db_path: str = DB_PATH):
    if username == "admin":
        return False, "Cannot delete the built-in admin account."
    init_auth_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    return True, "User deleted."


def change_password(username: str, new_password: str, db_path: str = DB_PATH):
    init_auth_db(db_path)
    salt_hex, hash_hex = _hash_password(new_password)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE users SET salt = ?, password_hash = ? WHERE username = ?",
        (salt_hex, hash_hex, username),
    )
    conn.commit()
    conn.close()
