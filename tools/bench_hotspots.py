#!/usr/bin/env python3
"""A/B the scan hot spots that were rewritten, against a live catalog.

Each pair runs the *old* strategy and the *new* one over the same rows in the
same database, so the ratio isolates the change. Numbers are hardware
specific — run it on your workstation and cite those.

    python tools/bench_hotspots.py --data DIR --state DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/user_shape")
    ap.add_argument("--state", default="/tmp/kpviz_shape")
    ap.add_argument("--spill-rows", type=int, default=60_000)
    ap.add_argument("--spill-files", type=int, default=40)
    a = ap.parse_args()

    from kpviz.config import init_settings
    st = init_settings(a.data, state_dir=a.state)
    from kpviz import db, derive
    from kpviz.scanner import _cost_var_sums

    con = db.connect()
    n_preds = db.q1("SELECT count(*) FROM preds")[0]
    n_runs = db.q1("SELECT count(*) FROM runs")[0]
    n_files = db.q1("SELECT count(*) FROM files")[0]
    print(f"catalog: {n_preds:,} preds · {n_runs} runs · {n_files:,} files\n")
    rows_out = []

    # ---- 1. run cost aggregation ----------------------------------------
    def old_costs():
        out = {}
        for ds, m, ar, rid, cjson in db.q(
                """SELECT dataset, model, arch, run_id, costs FROM preds
                   WHERE costs IS NOT NULL"""):
            try:
                cd = json.loads(cjson)
            except Exception:
                continue
            slot = out.setdefault((ds, m, ar, rid), {})
            for var, val in (cd or {}).items():
                if isinstance(val, (int, float)):
                    s, n = slot.get(var, (0.0, 0))
                    slot[var] = (s + float(val), n + 1)
        return out

    t_old, o = timed(old_costs)
    t_new, nn = timed(lambda: _cost_var_sums("preds"))
    same = all(abs(o[k].get(v, (0, 0))[0] - nn[k].get(v, (0, 0))[0]) < 1e-6
               and o[k].get(v, (0, 0))[1] == nn[k].get(v, (0, 0))[1]
               for k in set(o) & set(nn) for v in set(o[k]) | set(nn[k]))
    rows_out.append(("cost aggregation over preds (python json.loads -> SQL)",
                     t_old, t_new, f"identical results: {same}"))

    # ---- 2. spill ingest strategy ---------------------------------------
    tmp = st.tmp_dir / "bench"
    tmp.mkdir(parents=True, exist_ok=True)
    per = max(1, a.spill_rows // a.spill_files)
    paths = []
    for i in range(a.spill_files):
        rows = [{"kp": f"phrase {i}-{j}", "raw": f"Phrase {i}-{j}",
                 "lang": "en", "tokens": ["phrase", str(j)],
                 "stems": ["phrase", str(j)], "n_tokens": 2}
                for j in range(per)]
        p = tmp / f"spill_{i}.ndjson"
        derive._write_ndjson(p, rows)
        paths.append(str(p))
    cols = {"kp": "VARCHAR", "raw": "VARCHAR", "lang": "VARCHAR",
            "tokens": "VARCHAR[]", "stems": "VARCHAR[]", "n_tokens": "INTEGER"}

    con.execute("DROP TABLE IF EXISTS bench_pk")
    con.execute("""CREATE TABLE bench_pk(kp VARCHAR PRIMARY KEY, raw VARCHAR,
                   lang VARCHAR, tokens VARCHAR[], stems VARCHAR[],
                   n_tokens INTEGER)""")
    con.execute("DROP TABLE IF EXISTS bench_plain")
    con.execute("""CREATE TABLE bench_plain(kp VARCHAR, raw VARCHAR,
                   lang VARCHAR, tokens VARCHAR[], stems VARCHAR[],
                   n_tokens INTEGER)""")
    collist = ", ".join(cols)
    colspec = ", ".join(f"'{c}': '{t}'" for c, t in cols.items())

    def old_ingest():
        for p in paths:
            con.execute(f"INSERT OR REPLACE INTO bench_pk ({collist}) "
                        f"SELECT {collist} FROM read_json(?, "
                        f"format='newline_delimited', columns={{{colspec}}})", [p])

    def new_ingest():
        con.execute(f"INSERT INTO bench_plain ({collist}) "
                    f"SELECT {collist} FROM read_json(?, "
                    f"format='newline_delimited', columns={{{colspec}}})", [paths])

    t_old, _ = timed(old_ingest)
    t_new, _ = timed(new_ingest)
    n_pk = con.execute("SELECT count(*) FROM bench_pk").fetchone()[0]
    n_pl = con.execute("SELECT count(*) FROM bench_plain").fetchone()[0]
    rows_out.append((f"ingest {n_pl:,} rows in {a.spill_files} spills "
                     "(per-file upsert on PK -> one INSERT over the list)",
                     t_old, t_new, f"rows {n_pk:,} vs {n_pl:,}"))

    # ---- 3. DELETE against a PK index vs a plain table -------------------
    t_old, _ = timed(lambda: con.execute("DELETE FROM bench_pk"))
    t_new, _ = timed(lambda: con.execute("DELETE FROM bench_plain"))
    rows_out.append(("DELETE the same rows (PK index -> no index)",
                     t_old, t_new, "purge-then-insert keeps uniqueness"))

    # ---- 4. row-at-a-time DML vs one bulk statement ----------------------
    con.execute("DROP TABLE IF EXISTS bench_files")
    con.execute("CREATE TABLE bench_files(relpath VARCHAR PRIMARY KEY, "
                "file_id BIGINT, size BIGINT)")
    sample = db.q(f"SELECT relpath, file_id, size FROM files LIMIT 5000")

    def old_rows():
        for rel, fid, size in sample:
            con.execute("SELECT nextval('seq_file_id')")
            con.execute("INSERT OR REPLACE INTO bench_files VALUES (?,?,?)",
                        [rel, fid, size])

    def new_rows():
        p = tmp / "files.ndjson"
        derive._write_ndjson(p, [{"relpath": r, "file_id": f, "size": s}
                                 for r, f, s in sample])
        con.execute("INSERT OR REPLACE INTO bench_files (relpath, file_id, size) "
                    "SELECT relpath, file_id, size FROM read_json(?, "
                    "format='newline_delimited', columns={'relpath':'VARCHAR',"
                    "'file_id':'BIGINT','size':'BIGINT'})", [str(p)])

    t_old, _ = timed(old_rows)
    con.execute("DELETE FROM bench_files")
    t_new, _ = timed(new_rows)
    rows_out.append((f"register {len(sample):,} files (2 statements each -> 1 bulk)",
                     t_old, t_new, ""))

    for t in ("bench_pk", "bench_plain", "bench_files"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    for p in tmp.glob("*"):
        p.unlink()

    w = max(len(r[0]) for r in rows_out)
    print(f"{'change'.ljust(w)}   {'before':>10} {'after':>10} {'speedup':>9}")
    print("-" * (w + 34))
    for name, told, tnew, note in rows_out:
        sp = (told / tnew) if tnew > 0 else float("inf")
        print(f"{name.ljust(w)}   {told * 1000:9.1f}ms {tnew * 1000:9.1f}ms "
              f"{sp:8.1f}x  {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
