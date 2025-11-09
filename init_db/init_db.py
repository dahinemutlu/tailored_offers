#!/usr/bin/env python3
"""
init_db.py
- Drops existing tables (auto_tag_statistic, client_tag, tag_config)
- Recreates schema from schema.sql
- Loads CSV data from ./data/
  * tag_config.csv includes an explicit 'id' column
  * client_tag.csv uses numeric tag_id values
  * auto_tag_statistic.csv contains daily roll-ups for automatic tags
- Resets the identity sequence on tag_config.id after import
"""

import sys
import pathlib
import psycopg
import tomllib  # Python 3.11+. For Python 3.10, install 'tomli' and: import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SCHEMA_PATH = ROOT / "schema.sql"
CONFIG_PATH = ROOT / "db_url.toml"

TAG_CONFIG_CSV = DATA_DIR / "tag_config.csv"           # includes 'id'
CLIENT_TAG_CSV = DATA_DIR / "client_tag.csv"
AUTO_TAG_STAT_CSV = DATA_DIR / "auto_tag_statistic.csv"

# Drop order respects FKs: child tables first, then parents.
DROP_SQL = """
DROP TABLE IF EXISTS auto_tag_statistic;
DROP TABLE IF EXISTS client_tag;
DROP TABLE IF EXISTS tag_config;
"""

def load_db_url() -> str:
    with CONFIG_PATH.open("rb") as f:
        cfg = tomllib.load(f)
    try:
        return cfg["database"]["url"]
    except KeyError as e:
        raise SystemExit("❌ Missing [database].url in db_url.toml") from e

def copy_file(cur, table: str, cols_csv: str, path: pathlib.Path) -> None:
    with cur.copy(f"""
        COPY {table}
        ({cols_csv})
        FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
    """) as copy:
        with path.open("rb") as f:
            copy.write(f.read())

def main() -> None:
    # Basic existence checks
    for p in (TAG_CONFIG_CSV, CLIENT_TAG_CSV, AUTO_TAG_STAT_CSV):
        if not p.exists():
            raise SystemExit(f"❌ Missing required CSV: {p}")

    db_url = load_db_url()

    print("🔌 Connecting to database…")
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # 1) Drop
            print("🗑️  Dropping existing tables (if any)…")
            cur.execute(DROP_SQL)

            # 2) Schema
            print("📦 Applying schema.sql…")
            schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
            cur.execute(schema_sql)

            # 3) tag_config
            print("📥 Importing data: tag_config.csv (with id)…")
            copy_file(
                cur,
                "tag_config",
                "id, system_name, display_name, tag_type, color, description, is_active, created_at, updated_at",
                TAG_CONFIG_CSV
            )

            # 3b) Reset identity sequence for tag_config.id so next insert is MAX(id)+1
            print("🔧 Resetting identity sequence for tag_config.id…")
            cur.execute("""
                SELECT setval(
                    pg_get_serial_sequence('tag_config', 'id'),
                    COALESCE((SELECT MAX(id) FROM tag_config), 0)
                );
            """)

            # 4) client_tag (expects valid numeric tag_id values)
            print("📥 Importing data: client_tag.csv…")
            try:
                copy_file(
                    cur,
                    "client_tag",
                    "client_id, ont_id, tag_id, assigned_at, assigned_by, reason",
                    CLIENT_TAG_CSV
                )
            except Exception as e:
                raise SystemExit(
                    "❌ Failed to import client_tag.csv.\n"
                    "Check that all tag_id values exist in tag_config.id.\n"
                    f"Original error: {e}"
                )

            # 5) auto_tag_statistic
            # Columns: tag_id, assigned_count, run_started_at, run_finished_at
            print("📥 Importing data: auto_tag_statistic.csv…")
            try:
                copy_file(
                    cur,
                    "auto_tag_statistic",
                    "tag_id, assigned_count, run_started_at, run_finished_at",
                    AUTO_TAG_STAT_CSV
                )
            except Exception as e:
                raise SystemExit(
                    "❌ Failed to import auto_tag_statistic.csv.\n"
                    "Ensure tag_id values exist in tag_config.id and timestamps are valid.\n"
                    f"Original error: {e}"
                )

        conn.commit()

    print("✅ Database initialized and seeded (including auto_tag_statistic).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)
