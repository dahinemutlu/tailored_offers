#!/usr/bin/env python3
"""
init_db.py
- Drops existing tables (auto_tag_statistic, client_tag, tag_config, client)
- Recreates schema from schema.sql
- Loads CSV data from ./data/:
  * tag_config.csv (includes 'id')  [required]
  * client.csv                      [optional]
  * client_tag.csv                  [required]
  * auto_tag_statistic.csv          [required]
- Resets the identity sequence on tag_config.id after import
"""

import sys
import pathlib
import psycopg
import tomllib  # Python 3.11+. For Python 3.10: pip install tomli && `import tomli as tomllib`

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SCHEMA_PATH = ROOT / "schema.sql"
CONFIG_PATH = ROOT / "db_url.toml"

TAG_CONFIG_CSV    = DATA_DIR / "tag_config.csv"          # includes 'id'
CLIENT_CSV        = DATA_DIR / "client.csv"              # optional
CLIENT_TAG_CSV    = DATA_DIR / "client_tag.csv"          # required
AUTO_TAG_STAT_CSV = DATA_DIR / "auto_tag_statistic.csv"  # required

# Drop children -> parents (no FKs to client, but keep order tidy)
DROP_SQL = """
DROP TABLE IF EXISTS auto_tag_statistic;
DROP TABLE IF EXISTS client_tag;
DROP TABLE IF EXISTS tag_config;
DROP TABLE IF EXISTS client;
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
    # Required files
    if not SCHEMA_PATH.exists():
        raise SystemExit(f"❌ Missing schema.sql at {SCHEMA_PATH}")
    if not TAG_CONFIG_CSV.exists():
        raise SystemExit(f"❌ Missing required CSV: {TAG_CONFIG_CSV}")
    if not CLIENT_TAG_CSV.exists():
        raise SystemExit(f"❌ Missing required CSV: {CLIENT_TAG_CSV}")
    if not AUTO_TAG_STAT_CSV.exists():
        raise SystemExit(f"❌ Missing required CSV: {AUTO_TAG_STAT_CSV}")

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

            # 3) Optional client load (no FK dependency)
            if CLIENT_CSV.exists():
                print("📥 Importing data: client.csv…")
                copy_file(
                    cur,
                    "client",
                    "client_id, ont_id, name, phone, service_id, city, area, address, type, sip",
                    CLIENT_CSV
                )
            else:
                print("ℹ️  Skipping client.csv (not found). No FK requires it right now.")

            # 4) tag_config (with explicit id)
            print("📥 Importing data: tag_config.csv (with id)…")
            copy_file(
                cur,
                "tag_config",
                "id, system_name, display_name, tag_type, color, description, is_active, created_at, updated_at",
                TAG_CONFIG_CSV
            )

            # 4b) reset identity for tag_config.id
            print("🔧 Resetting identity sequence for tag_config.id…")
            cur.execute("""
                SELECT setval(
                    pg_get_serial_sequence('tag_config', 'id'),
                    COALESCE((SELECT MAX(id) FROM tag_config), 0)
                );
            """)

            # 5) client_tag
            print("📥 Importing data: client_tag.csv…")
            copy_file(
                cur,
                "client_tag",
                "client_id, ont_id, tag_id, assigned_at, assigned_by, reason",
                CLIENT_TAG_CSV
            )

            # 6) auto_tag_statistic
            print("📥 Importing data: auto_tag_statistic.csv…")
            copy_file(
                cur,
                "auto_tag_statistic",
                "tag_id, assigned_count, run_started_at, run_finished_at",
                AUTO_TAG_STAT_CSV
            )

        conn.commit()

    print("✅ Database initialized and seeded (client optional).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
