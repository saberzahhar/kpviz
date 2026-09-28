#!/usr/bin/env sh
# Launch KPViz: find Python >= 3.11, create .venv, install, serve.
#
#   ./run.sh                     serves ./data, else ./sample_data (generated if missing)
#   ./run.sh --data /path/to/data --port 8050
#
# Arguments are passed to the app (./run.sh --help lists them).
# KPVIZ_PYTHON=/path/to/python selects the interpreter.
set -eu
cd "$(dirname "$0")"

fail() { echo "error: $*" >&2; exit 1; }

# 1. Python >= 3.11
ok() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; }
PY=""
for c in ${KPVIZ_PYTHON:-} python3.13 python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && ok "$c"; then PY=$(command -v "$c"); break; fi
done
[ -n "$PY" ] || fail "KPViz needs Python 3.11 or newer (https://www.python.org/downloads/)."

# 2. virtual environment
VENV=.venv
if ! { ok "$VENV/bin/python" || ok "$VENV/Scripts/python.exe"; }; then
  echo "· creating $VENV with $("$PY" --version)"
  rm -rf "$VENV"
  "$PY" -m venv "$VENV" || fail "could not create $VENV (on Debian/Ubuntu: apt install python3-venv)"
fi
VPY="$VENV/bin/python"
[ -x "$VPY" ] || VPY="$VENV/Scripts/python.exe"        # Git Bash on Windows

# 3. dependencies, installed again only when the dependency files change
STAMP="$VENV/.kpviz-deps"
SUM=$(cat pyproject.toml constraints.txt | cksum)
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$SUM" ]; then
  echo "· installing KPViz and its dependencies (first run: a few minutes)"
  "$VPY" -m pip install -q --upgrade pip
  "$VPY" -m pip install -q -r requirements.txt -c constraints.txt
  echo "$SUM" > "$STAMP"
fi

# 4. data: ./data if present, else the generated sample tree
case " $* " in
  *" --data "*|*" --data="*) ;;
  *) if [ ! -d data ] && [ ! -d sample_data ]; then
       echo "· no ./data: generating sample_data/"
       "$VPY" tools/make_sample_data.py
     fi ;;
esac

# 5. serve
exec "$VPY" app.py "$@"
