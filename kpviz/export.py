"""Export engine: PGF / PDF / PNG figures and booktabs LaTeX tables.

PGF needs a TeX distribution (it typesets labels with your document's
fonts). When none is installed we degrade explicitly: PDF falls back to
Matplotlib's own vector backend and the LaTeX snippet switches to
\\includegraphics — nothing breaks, the caption says what happened.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import threading
from collections import OrderedDict

from .figures import _TEX_TABLE, render
from .util import fmt_num, stable_hash

# Matplotlib's pyplot figure manager and rcParams are process-global, and the
# server is threaded: two concurrent exports would interleave and corrupt each
# other's output. One render at a time, with a small result cache so repeated
# downloads of the same figure are free.
_render_lock = threading.RLock()
_CACHE: "OrderedDict[str, bytes | str]" = OrderedDict()
_CACHE_BYTES = 64 * 2**20
_cache_size = [0]
_FAILED: dict[str, str] = {}          # render key -> error (never retried)


def _render_key(kind: str, spec: dict) -> str:
    """What the rendered bytes depend on. The caption and the LaTeX table
    are not drawn in the figure: editing a caption must not re-render (or
    evict) the figure."""
    drawn = {k: v for k, v in spec.items() if k not in ("caption", "table")}
    return kind + ":" + stable_hash(drawn)


def _cached(kind: str, spec: dict, build):
    key = _render_key(kind, spec)
    with _render_lock:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
        val = build()
        if val is not None:
            _CACHE[key] = val
            _cache_size[0] += len(val)
            while _cache_size[0] > _CACHE_BYTES and len(_CACHE) > 1:
                _k, old = _CACHE.popitem(last=False)
                _cache_size[0] -= len(old)
        return val


# The probe must be as hard as the real thing. A bare "x"/"y" figure compiles
# even on a TeX install whose lualatex cannot load fonts (luaotfload missing →
# "reverting to OT1"), and the engine is then certified for exports that fail —
# silently, since fig_pdf falls back. So the probe carries what real captions
# carry: an em dash, a dagger, an underscore, Greek, a superscript and math.
# The log axis is not decoration: its 10^-6 ticks are the only *math* in a
# KPViz figure, and a TeX install can typeset text yet fail to load a math
# font ("Font \TU/lmr/… not loadable: metric data not found"), which is
# exactly the failure a text-only probe misses.
_PROBE_SPEC = {
    "kind": "scatter", "size": "1col", "xscale": "log",
    "xlabel": "cost (USD) — per document, log scale",
    "ylabel": "F1@O — α ± 0.02 †",
    "series": [{"name": "probe", "x": [1e-6, 1e-3], "y": [0.0, 1.0],
                "color": "#2a78d6", "mpl_marker": "o", "in_legend": False,
                "text": ["bart-base_kp20k (num_beams=4) †", ""],
                "show_text": True}],
    "hlines": [{"y": 0.5, "label": "no usd — 100 %", "dash": True}],
    "frontier": {"x": [1e-6, 1e-3], "y": [0.0, 1.0]},
}
# pdflatex first: it is the fastest engine and the one most papers use;
# lualatex (slowest, and the one most often missing fonts) last
_ENGINES = ("pdflatex", "xelatex", "lualatex")

_PROBE_NOTES: list[str] = []
_probe = {"status": "idle", "engine": None}    # idle | probing | ready
_probe_lock = threading.Lock()
_demoted: set[str] = set()


def _probe_key() -> dict:
    import matplotlib
    key = {"mpl": matplotlib.__version__, "engines": {}}
    for eng in _ENGINES:
        path = shutil.which(eng)
        if not path:
            continue
        try:
            ver = subprocess.run([path, "--version"], capture_output=True,
                                 text=True, timeout=10).stdout.splitlines()[:1]
        except Exception:
            ver = []
        key["engines"][eng] = [path, ver[0] if ver else ""]
    return key


def _probe_file():
    try:
        from .config import settings
        return settings().state_dir / "tex_probe.json"
    except Exception:
        return None


def _run_probe() -> str | None:
    """First engine that compiles a *representative* PGF figure. The result
    is persisted per (engine binaries, versions, Matplotlib), so a restart
    does not pay for TeX again."""
    import matplotlib
    key = _probe_key()
    pf = _probe_file()
    if pf and pf.exists():
        try:
            saved = json.loads(pf.read_text())
            if saved.get("key") == key:
                _PROBE_NOTES[:] = saved.get("notes", [])
                return saved.get("engine")
        except Exception:
            pass
    engine = None
    notes: list[str] = []
    for eng in _ENGINES:
        if eng not in key["engines"]:
            continue
        try:
            with _render_lock:
                matplotlib.rcParams["pgf.texsystem"] = eng
                matplotlib.rcParams["pgf.preamble"] = ""
                # the exact export path (tight bbox measures text via TeX)
                pdf = render(_PROBE_SPEC, True, "pdf", backend="pgf",
                             bbox_inches="tight")
            if pdf[:5] == b"%PDF-":
                engine = eng
                break
            notes.append(f"{eng}: produced no PDF")
        except Exception as e:
            first = next((ln for ln in str(e).splitlines()
                          if ln.strip().startswith("!")), "").strip()
            notes.append(f"{eng}: {first or type(e).__name__}")
    _PROBE_NOTES[:] = notes
    if engine is None and notes:
        print("· no usable TeX engine — exports use Matplotlib's vector "
              "backend.\n  " + "\n  ".join(notes))
    if pf:
        try:
            pf.write_text(json.dumps({"key": key, "engine": engine,
                                      "notes": notes}))
        except OSError:
            pass
    return engine


def start_tex_probe() -> None:
    """Probe TeX once, in the background, at start-up — never inside a
    callback (six clipboard callbacks used to each run it at first load)."""
    with _probe_lock:
        if _probe["status"] != "idle":
            return
        _probe["status"] = "probing"

    def run():
        eng = None
        try:
            eng = _run_probe()
        finally:
            with _probe_lock:
                _probe.update(status="ready", engine=eng)
    threading.Thread(target=run, daemon=True, name="kpviz-tex-probe").start()


def tex_status() -> tuple[str, str | None]:
    """('probing' | 'ready', engine or None) — never blocks."""
    with _probe_lock:
        eng = _probe["engine"]
        if eng in _demoted:
            eng = None
        return _probe["status"], eng


def tex_engine(wait: float = 120.0) -> str | None:
    """The usable engine, waiting for a running probe (exports only)."""
    import time
    if _probe["status"] == "idle":
        start_tex_probe()
    t0 = time.time()
    while tex_status()[0] == "probing" and time.time() - t0 < wait:
        time.sleep(0.05)
    return tex_status()[1]


def _configure_pgf(eng: str):
    import matplotlib
    matplotlib.rcParams["pgf.texsystem"] = eng
    matplotlib.rcParams["pgf.preamble"] = ""


def _first_tex_error(exc: Exception) -> str:
    first = next((ln for ln in str(exc).splitlines()
                  if ln.strip().startswith("!")), "").strip()
    return first or f"{type(exc).__name__}: {str(exc)[:160]}"


def _demote(eng: str, err: str) -> None:
    """A real render failed although the probe passed: stop offering this
    engine for PGF (reported once)."""
    if eng not in _demoted:
        _demoted.add(eng)
        print(f"· {eng} failed a real PGF render ({err}); PGF disabled for "
              f"this session, PDF falls back to Matplotlib")


def fig_png(spec: dict, dpi: int = 300) -> bytes:
    def build():
        with _render_lock:
            return render(spec, False, "png", dpi=dpi, facecolor="white",
                          bbox_inches="tight", pad_inches=0.02)
    return _cached(f"png{dpi}", spec, build)


def fig_pdf(spec: dict) -> tuple[bytes, str, str]:
    """(pdf bytes, method, note): 'pgf' (TeX-typeset) or 'matplotlib'
    (vector fallback with embedded TrueType fonts, no TeX required)."""
    note = [""]

    def build():
        eng = tex_engine()
        with _render_lock:
            if eng:
                try:
                    _configure_pgf(eng)
                    return b"pgf:" + render(spec, True, "pdf", backend="pgf",
                                            bbox_inches="tight", pad_inches=0.02)
                except Exception as e:
                    err = _first_tex_error(e)
                    _demote(eng, err)
                    note[0] = f"{eng} failed ({err})"
            return b"mpl:" + render(spec, False, "pdf", bbox_inches="tight",
                                    pad_inches=0.02)
    blob = _cached("pdf", spec, build)
    tag, _, payload = blob.partition(b":")
    return payload, ("TeX (PGF)" if tag == b"pgf" else "Matplotlib"), note[0]


def fig_pgf(spec: dict) -> tuple[str | None, str]:
    """(.pgf source, "") or (None, reason). Failures are remembered, so a
    broken figure does not re-run TeX on every click."""
    eng = tex_engine()
    if not eng:
        return None, "no working TeX engine on this machine"
    key = _render_key(f"pgf-{eng}", spec)
    if key in _FAILED:
        return None, _FAILED[key]

    def build():
        with _render_lock:
            try:
                _configure_pgf(eng)
                return render(spec, True, "pgf", backend="pgf").decode("utf-8")
            except Exception as e:
                _FAILED[key] = f"{eng}: {_first_tex_error(e)}"
                return None
    out = _cached(f"pgf-{eng}", spec, build)
    return (out, "") if out is not None else (None, _FAILED.get(key, "failed"))


# ---------------------------------------------------------------------------
# LaTeX snippets
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    """ASCII specials first, then the non-ASCII → TeX translation (whose output
    is itself TeX: escaping after it would break `$\\ddagger$` into text)."""
    s = str(s)
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    s = re.sub(r"[\\&%$#_{}~^]", lambda m: repl[m.group(0)], s)
    # "<" and ">" are text-mode specials under OT1 (they render as ¡ and ¿);
    # every cell here is machine-generated, so math-mode them unconditionally.
    s = s.replace("<", "$<$").replace(">", "$>$")
    if not s.isascii():
        s = s.translate(_TEX_TABLE)
    # a significance mark attached to a number is a superscript, as in the UI
    return re.sub(r"(?<=[\d)])\s*\$\\(dagger|ddagger)\$",
                  lambda m: "$^{\\" + m.group(1) + "}$", s)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "figure").lower()).strip("-")[:48]


def caption_escape(s: str) -> str:
    """Escape the LaTeX specials that commonly break captions (& % # and a
    bare _), while leaving backslash commands like \\emph{} intact. Then
    translate the non-ASCII glyphs a caption really uses — em dashes, ≥, the
    significance daggers — into TeX, so the pasted caption typesets under any
    engine instead of dropping characters."""
    s = re.sub(r"(?<!\\)([&%#])", r"\\\1", str(s))
    s = re.sub(r"(?<!\\)_", r"\\_", s)
    # "p<0.05" must not become "p¡0.05" under OT1 — but the caption is editable,
    # so only touch segments outside the user's own $…$ math.
    parts = s.split("$")
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("<", "$<$").replace(">", "$>$")
    s = "$".join(parts)
    return s if s.isascii() else s.translate(_TEX_TABLE)


# Matplotlib writes \mathdefault into every .pgf but defines it only in the
# preamble it uses internally, so an \input-ed figure dies with "Undefined
# control sequence" in a real paper. \providecommand is idempotent, so shipping
# it with each figure keeps the snippet self-contained and safe to repeat.
PGF_PREAMBLE_NOTE = ("% requires \\usepackage{pgf} in your preamble\n"
                     "\\providecommand{\\mathdefault}[1]{#1}")


def latex_figure(filename: str, caption: str, label: str,
                 width: str | None = None, pgf: bool = True,
                 size: str = "2col") -> str:
    """A figure environment for the exported file.

    The PGF is \\input at its natural size — never \\resizebox-ed, which
    scaled the typeset labels with it (a 7 in figure squeezed into a 3.3 in
    column set 7 pt text at ~3 pt). A two-column figure ("2col", 7 in) goes
    into `figure*`, spanning \\textwidth; a one-column figure (3.35 in) into
    `figure`. The PDF fallback is scaled to the same width, which is harmless
    for an image."""
    env = "figure*" if size in ("2col", "slide") else "figure"
    width = width or ("\\textwidth" if env == "figure*" else "\\columnwidth")
    body = (f"    \\input{{{filename}}}" if pgf
            else f"    \\includegraphics[width={width}]{{{filename}}}")
    lines = [
        PGF_PREAMBLE_NOTE if pgf else None,
        f"\\begin{{{env}}}[t]",
        "    \\centering",
        body,
        f"    \\caption{{{caption_escape(caption)}}}",
        f"    \\label{{fig:{label}}}",
        f"\\end{{{env}}}",
    ]
    return "\n".join(l for l in lines if l is not None)


# A column header like "document truncated to model's context window" makes a
# five-column tabular wider than \textwidth: LaTeX then runs it off the page
# with only an Overfull \hbox warning. Past this printable width the tabular is
# wrapped in \resizebox, which is what a human would do by hand.
_TABLE_FIT_CHARS = 78


def latex_table(headers: list[str], rows: list[list], caption: str,
                label: str, align: str | None = None,
                best_mask: list[list[bool]] | None = None) -> str:
    """A booktabs table. best_mask marks cells to \\textbf{} (e.g. column
    maxima)."""
    ncol = len(headers)
    align = align or ("l" + "r" * (ncol - 1))
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for j, cell in enumerate(row[:ncol]):
            widths[j] = max(widths[j], len(str(cell)))
    wide = sum(widths) + 2 * ncol > _TABLE_FIT_CHARS
    out = [
        "\\begin{table}[t]",
        "    \\centering",
        f"    \\caption{{{caption_escape(caption)}}}",
        f"    \\label{{tab:{label}}}",
        "    \\setlength{\\tabcolsep}{4pt}",
    ]
    if wide:
        out.append("    \\resizebox{\\linewidth}{!}{%")
    out += [
        f"    \\begin{{tabular}}{{{align}}}",
        "        \\toprule",
        "        " + " & ".join(_tex_escape(h) for h in headers) + " \\\\",
        "        \\midrule",
    ]
    for i, row in enumerate(rows):
        cells = []
        for j, cell in enumerate(row):
            txt = _tex_escape(cell if isinstance(cell, str) else fmt_num(cell))
            if best_mask and best_mask[i][j]:
                txt = "\\textbf{" + txt + "}"
            cells.append(txt)
        out.append("        " + " & ".join(cells) + " \\\\")
    out += ["        \\bottomrule", "    \\end{tabular}"]
    if wide:
        out.append("    }")
    out.append("\\end{table}")
    return "\n".join(out)


def export_bundle(spec: dict, name: str) -> bytes:
    """Zip with fig.pdf (+ fig.pgf when TeX available) + fig.png + fig.tex."""
    import zipfile
    slug = slugify(name)
    pdf, method, _note = fig_pdf(spec)
    pgf, _err = fig_pgf(spec)
    png = fig_png(spec)
    caption = spec.get("caption", "")
    tex = latex_figure(f"{slug}.pgf" if pgf else f"{slug}.pdf",
                       caption, slug, pgf=bool(pgf),
                       size=spec.get("size", "2col"))
    tab = spec.get("table")
    table_tex = (latex_table(tab["headers"], tab["rows"], caption,
                             tab.get("label", slug)) if tab else None)
    buf = io.BytesIO()
    # PDF/PNG are already compressed; level 1 keeps the text members small
    # without spending CPU re-compressing binary ones
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.writestr(f"{slug}.pdf", pdf)
        if pgf:
            z.writestr(f"{slug}.pgf", pgf)
        z.writestr(f"{slug}.png", png)
        z.writestr(f"{slug}.tex", tex)
        if table_tex:
            z.writestr(f"{slug}-table.tex", table_tex)
        z.writestr("figure.json", json.dumps(spec, ensure_ascii=False, indent=1,
                                             default=str))
        z.writestr("README.txt",
                   f"KPViz export — {name}\n"
                   f"PDF rendered via: {method}\n"
                   + ("PGF included - drop " + slug + ".tex into your paper "
                      "(it \\input{}s the .pgf); fonts follow your document.\n"
                      "Preamble: \\usepackage{pgf}  (the .tex already carries "
                      "\\providecommand{\\mathdefault}[1]{#1}, which Matplotlib\n"
                      "writes into every .pgf but never defines).\n"
                      "Verified to compile with pdflatex, xelatex and lualatex.\n"
                      if pgf else
                      "No TeX distribution found at export time - "
                      "PGF omitted, use the PDF via \\includegraphics.\n")
                   + f"Caption:\n{caption}\n")
    return buf.getvalue()
