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

from .figures import _TEX_TABLE, VENUES, geometry, render
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


def _cached(kind: str, spec: dict, build, store: bool = True):
    """Render once per key. `build` runs under the render lock, so it must
    never wait on anything that itself needs the lock (the TeX probe):
    callers resolve the engine *before* calling this and put it in `kind`.
    store=False renders without caching (a provisional result)."""
    key = _render_key(kind, spec)
    with _render_lock:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
        val = build()
        if val is not None and store:
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


def _close_latex_managers() -> None:
    """Close Matplotlib's cached LaTeX subprocesses before interpreter
    shutdown. Its own weakref finalizer runs after stdin is closed and
    prints "ValueError: I/O operation on closed file" on every exit of a
    process that rendered PGF (the error type it guards against is
    RuntimeError). Running each finalizer here, guarded, marks it done."""
    import sys
    if "matplotlib.backends.backend_pgf" not in sys.modules:
        return
    import gc
    from matplotlib.backends.backend_pgf import LatexManager
    for obj in gc.get_objects():
        if isinstance(obj, LatexManager):
            for name in ("_finalize_latex", "_finalize_tmpdir"):
                fin = getattr(obj, name, None)
                try:
                    if fin is not None and fin.alive:
                        fin()
                except Exception:
                    pass


import atexit as _atexit                       # noqa: E402
_atexit.register(_close_latex_managers)


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
    """The usable engine, waiting for a running probe (exports only).

    Never call this while holding `_render_lock`: the probe renders under
    that lock, so a waiter holding it would stall until the timeout."""
    import time
    if _probe["status"] == "idle":
        start_tex_probe()
    t0 = time.time()
    while tex_status()[0] == "probing" and time.time() - t0 < wait:
        time.sleep(0.05)
    return tex_status()[1]


def _engines_after(eng: str) -> list[str]:
    """Installed engines to try on a figure `eng` could not typeset."""
    out = []
    for e in _ENGINES:
        if e != eng and e not in _demoted and shutil.which(e):
            out.append(e)
    return out


_KPSE: dict[str, bool] = {}


def _have_sty(name: str) -> bool:
    """Is `name`.sty installed? (kpsewhich, cached)."""
    if name not in _KPSE:
        path = shutil.which("kpsewhich")
        ok = False
        if path:
            try:
                ok = bool(subprocess.run([path, f"{name}.sty"], capture_output=True,
                                         text=True, timeout=10).stdout.strip())
            except Exception:
                ok = False
        _KPSE[name] = ok
    return _KPSE[name]


def venue_preamble(spec: dict) -> tuple[str, str]:
    """(preamble, note) that typesets a KPViz-compiled PDF in the venue's
    fonts, falling back to the closest installed packages."""
    exp = spec.get("export") or {}
    v = VENUES.get(exp.get("venue") or "generic")
    if not v:
        return "", ""

    def usable(lines):
        pk = [re.search(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", ln) for ln in lines]
        return all(m and _have_sty(m.group(1)) for m in pk)
    if usable(v["preamble"]):
        return "\n".join(v["preamble"]), ""
    fb = v.get("fallback") or []
    if fb and usable(fb):
        return "\n".join(fb), (f"{v['label']} fonts are not installed here; the "
                               "PDF is set in Times (the .pgf takes your "
                               "document's fonts)")
    return "", (f"{v['label']} fonts are not installed here; the PDF uses "
                "Computer Modern (the .pgf takes your document's fonts)")


def _configure_pgf(eng: str, preamble: str = ""):
    import matplotlib
    matplotlib.rcParams["pgf.texsystem"] = eng
    matplotlib.rcParams["pgf.preamble"] = preamble


def _first_tex_error(exc: Exception) -> str:
    first = next((ln for ln in str(exc).splitlines()
                  if ln.strip().startswith("!")), "").strip()
    return first or f"{type(exc).__name__}: {str(exc)[:160]}"


def _demote(eng: str, err: str) -> None:
    """The engine itself broke (the probe figure no longer compiles either):
    stop offering it for this session (reported once)."""
    if eng not in _demoted:
        _demoted.add(eng)
        print(f"· {eng} no longer typesets the probe figure ({err}); PGF with "
              f"{eng} disabled for this session")


def _engine_still_works(eng: str) -> bool:
    """Re-run the probe figure with `eng` (caller holds the render lock).
    A failure on one figure demotes the engine only if this fails too — a
    label TeX cannot set is that figure's problem, not the engine's."""
    try:
        _configure_pgf(eng, "")
        return render(_PROBE_SPEC, True, "pdf", backend="pgf",
                      bbox_inches="tight")[:5] == b"%PDF-"
    except Exception:
        return False


def fig_png(spec: dict, dpi: int = 300) -> bytes:
    def build():
        with _render_lock:
            return render(spec, False, "png", dpi=dpi, facecolor="white",
                          bbox_inches="tight", pad_inches=0.02)
    return _cached(f"png{dpi}", spec, build)


def fig_pdf(spec: dict) -> tuple[bytes, str, str]:
    """(pdf bytes, method, note): 'TeX (PGF, <engine>)' or 'Matplotlib'
    (vector fallback with embedded TrueType fonts, no TeX required).

    The engine is resolved before the render lock is taken (the probe needs
    that lock) and is part of the cache key; a Matplotlib fallback made
    while the probe is still running is returned but never cached. A figure
    the engine cannot typeset is retried with the other installed engines;
    the engine is demoted only if the probe figure fails too."""
    pre, pre_note = venue_preamble(spec)
    eng = tex_engine()
    provisional = tex_status()[0] == "probing"
    note = [pre_note]
    key_pdf = _render_key(f"pdf-{eng or 'mpl'}", spec)

    def build():
        with _render_lock:
            tried = []
            if eng and key_pdf not in _FAILED:
                for e in [eng] + _engines_after(eng):
                    try:
                        _configure_pgf(e, pre)
                        pdf = render(spec, True, "pdf", backend="pgf",
                                     bbox_inches="tight", pad_inches=0.02)
                        if tried:
                            note[0] = "; ".join(tried) + f" — typeset with {e}"
                        return f"pgf-{e}:".encode() + pdf
                    except Exception as ex:
                        err = _first_tex_error(ex)
                        tried.append(f"{e} could not typeset this figure ({err})")
                        if not _engine_still_works(e):
                            _demote(e, err)
                _FAILED[key_pdf] = "; ".join(tried)
            if key_pdf in _FAILED:
                note[0] = _FAILED[key_pdf]
            return b"mpl:" + render(spec, False, "pdf", bbox_inches="tight",
                                    pad_inches=0.02)
    blob = _cached(f"pdf-{eng or 'mpl'}", spec, build, store=not provisional)
    tag, _, payload = blob.partition(b":")
    if tag == b"mpl" and key_pdf in _FAILED and not note[0]:
        note[0] = _FAILED[key_pdf]
    if tag.startswith(b"pgf-"):
        return payload, f"TeX (PGF, {tag[4:].decode()})", note[0]
    if provisional:
        note[0] = note[0] or "TeX check still running — Matplotlib PDF for now"
    return payload, "Matplotlib", note[0]


def fig_pgf(spec: dict) -> tuple[str | None, str]:
    """(.pgf source, "") or (None, reason). Failures are remembered, so a
    broken figure does not re-run TeX on every click."""
    eng = tex_engine()
    if not eng:
        return None, "no working TeX engine on this machine"
    key = _render_key(f"pgf-{eng}", spec)
    if key in _FAILED:
        return None, _FAILED[key]

    pre, _note = venue_preamble(spec)

    def build():
        with _render_lock:
            try:
                _configure_pgf(eng, pre)
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


FIG_DIR = "figures"


def export_name(spec: dict) -> tuple[str, str]:
    """(file stem, path the LaTeX snippet references) — the one naming rule
    for the copied snippet, the downloaded files and the bundle, so a
    copy-then-download workflow finds its files: `<name>[-<venue>]` saved
    into your paper's `figures/` folder."""
    stem = slugify(spec.get("name", "figure"))
    venue = (spec.get("export") or {}).get("venue")
    if venue and venue != "generic":
        stem = f"{stem}-{venue}"
    return stem, f"{FIG_DIR}/{stem}"


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


def figure_env(spec: dict) -> str:
    """`figure*` when the figure spans both columns of a two-column venue,
    `figure` otherwise (single-column venues, column-wide figures)."""
    exp = spec.get("export") or {}
    size = spec.get("size", "2col")
    if not exp:
        return "figure*" if size in ("2col", "slide") else "figure"
    v = VENUES.get(exp.get("venue") or "generic", VENUES["generic"])
    span = exp.get("span") or "auto"
    if span == "auto":
        span = "col" if size == "1col" else "full"
    return "figure*" if (v["twocol"] and span == "full") else "figure"


def latex_figure(filename: str, caption: str, label: str,
                 width: str | None = None, pgf: bool = True,
                 size: str = "2col", env: str | None = None,
                 dims: tuple | None = None) -> str:
    """A figure environment for the exported file.

    The PGF is \\input at its natural size — never \\resizebox-ed, which
    scaled the typeset labels with it (a 7 in figure squeezed into a 3.3 in
    column set 7 pt text at ~3 pt). A two-column figure ("2col", 7 in) goes
    into `figure*`, spanning \\textwidth; a one-column figure (3.35 in) into
    `figure`. The PDF fallback is scaled to the same width, which is harmless
    for an image."""
    env = env or ("figure*" if size in ("2col", "slide") else "figure")
    width = width or ("\\textwidth" if env == "figure*" else "\\columnwidth")
    body = (f"    \\input{{{filename}}}" if pgf
            else f"    \\includegraphics[width={width}]{{{filename}}}")
    lines = [
        PGF_PREAMBLE_NOTE if pgf else "% requires \\usepackage{graphicx} in your preamble",
        (f"% drawn at {dims[0]:.2f} x {dims[1]:.2f} in with {dims[2]:g} pt "
         "labels: \\input as is, never rescale" if dims and pgf else None),
        f"\\begin{{{env}}}[t]",
        "    \\centering",
        body,
        f"    \\caption{{{caption_escape(caption)}}}",
        f"    \\label{{fig:{label}}}",
        f"\\end{{{env}}}",
    ]
    return "\n".join(l for l in lines if l is not None)


TABLE_CELLS = ("value", "ci", "ci_n")      # how much of a stat cell to print


def _num_tex(v: float, digits: int, signed: bool) -> str:
    """A number for text mode: a real minus sign, never a hyphen."""
    txt = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    return txt.replace("-", "$-$", 1) if txt.startswith("-") else txt


def cell_text(cell, style: str = "ci") -> str:
    """Plain-text rendering of a table cell (structured or not) — what the
    width estimate and non-LaTeX consumers see."""
    if not isinstance(cell, dict):
        return str(cell)
    d, sg = cell.get("digits", 3), cell.get("signed", False)
    f = (lambda v: f"{v:+.{d}f}") if sg else (lambda v: f"{v:.{d}f}")
    out = f(cell["v"]) + (f" {cell['mark']}" if cell.get("mark") else "")
    if style in ("ci", "ci_n") and cell.get("lo") is not None:
        out += f" [{f(cell['lo'])}, {f(cell['hi'])}]"
    if style == "ci_n" and cell.get("n") is not None:
        out += f" (n={cell['n']})"
    return out


_SCI = re.compile(r"^(-?\d+(?:\.\d+)?)e([+-]?)0*(\d+)$")


def _cell_tex(cell, style: str) -> str:
    if not isinstance(cell, dict):
        txt = cell if isinstance(cell, str) else fmt_num(cell)
        m = _SCI.match(str(txt).strip())
        if m:                       # 9.13e-06 -> $9.13\times10^{-6}$
            exp = ("-" if m.group(2) == "-" else "") + m.group(3)
            return f"${m.group(1)}\\times10^{{{exp}}}$"
        return _tex_escape(txt)
    d, sg = cell.get("digits", 3), cell.get("signed", False)
    out = _num_tex(cell["v"], d, sg)
    if cell.get("mark"):
        out += "$^{" + "".join({"†": r"\dagger", "‡": r"\ddagger"}.get(ch, "")
                               for ch in cell["mark"]) + "}$"
    if style in ("ci", "ci_n") and cell.get("lo") is not None:
        out += (" {\\scriptsize[" + _num_tex(cell["lo"], d, sg) + ", "
                + _num_tex(cell["hi"], d, sg) + "]}")
    if style == "ci_n" and cell.get("n") is not None:
        out += f" {{\\scriptsize(n={cell['n']})}}"
    if cell.get("bold"):
        out = "\\textbf{" + out + "}"
    return out


def table_caption(spec: dict, caption: str | None = None) -> str:
    """A table's own caption: what the table lists, in one sentence, and a
    pointer to its figure — not the figure's whole caption again (in IEEE
    small caps that was six lines above every table). The statistical
    procedure goes in the table notes."""
    tab = spec.get("table") or {}
    if tab.get("caption"):
        return tab["caption"]
    cap = (caption or spec.get("caption") or "").strip()
    # first sentence: up to the first ". " that is not inside parentheses
    depth, cut = 0, len(cap)
    for i, ch in enumerate(cap):
        depth += (ch == "(") - (ch == ")")
        if ch == "." and depth == 0 and (i + 1 == len(cap) or cap[i + 1] == " "):
            cut = i + 1
            break
    first = cap[:cut].rstrip(".")
    slug = slugify(spec.get("name", "figure"))
    return (first + f". Values behind Figure~\\ref{{fig:{slug}}}"
            if first else f"Values behind Figure~\\ref{{fig:{slug}}}")


def latex_table(headers: list[str], rows: list[list], caption: str,
                label: str, align: str | None = None,
                best_mask: list[list[bool]] | None = None,
                cells: str = "ci", notes: str | None = None,
                venue: str | None = None) -> str:
    """A booktabs table, set in \\small.

    `cells` chooses how much of a statistic to print — the value (with its
    significance mark), + its interval, + its sample size; `notes` is a line
    under the rule (the statistical procedure, typically); best_mask marks
    cells to \\textbf{}.

    Width: a table wider than a column of a two-column venue goes into
    `table*` (it spans the page, as a human would place it); whatever the
    environment, adjustbox's `max width=\\linewidth` shrinks it only if it
    still does not fit — never a silent run into the margin, never a blow-up
    of a narrow table."""
    from .figures import VENUES
    ncol = len(headers)
    align = align or ("l" + "r" * (ncol - 1))
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for j, cell in enumerate(row[:ncol]):
            widths[j] = max(widths[j], len(cell_text(cell, cells)))
    v = VENUES.get(venue or "generic", VENUES["generic"])
    # \small digits are ~4.6 pt wide; 8 pt of padding per column
    need_in = (sum(widths) * 4.6 + ncol * 8) / 72
    env = "table*" if (v["twocol"] and need_in > v["col"]) else "table"
    out = [
        "% requires \\usepackage{booktabs} and \\usepackage{adjustbox}",
        f"\\begin{{{env}}}[t]",
        "    \\centering",
        "    \\small",
        f"    \\caption{{{caption_escape(caption)}}}",
        f"    \\label{{tab:{label}}}",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\begin{adjustbox}{max width=\\linewidth}",
        f"    \\begin{{tabular}}{{{align}}}",
        "        \\toprule",
        "        " + " & ".join(_tex_escape(h) for h in headers) + " \\\\",
        "        \\midrule",
    ]
    for i, row in enumerate(rows):
        cells_tex = []
        for j, cell in enumerate(row):
            txt = _cell_tex(cell, cells)
            if best_mask and best_mask[i][j]:
                txt = "\\textbf{" + txt + "}"
            cells_tex.append(txt)
        out.append("        " + " & ".join(cells_tex) + " \\\\")
    out += ["        \\bottomrule", "    \\end{tabular}", "    \\end{adjustbox}"]
    if notes:
        out += ["    \\par\\smallskip",
                "    {\\footnotesize " + caption_escape(notes) + "\\par}"]
    out.append(f"\\end{{{env}}}")
    return "\n".join(out)


def snippets(spec: dict, caption: str | None = None) -> tuple[str, str, str]:
    """(LaTeX figure, LaTeX table, hint) for a spec with its export options —
    never waiting on TeX (the probe result is read, not computed)."""
    caption = caption or spec.get("caption", "")
    slug = slugify(spec.get("name", "figure"))
    _stem, ref = export_name(spec)
    status, eng = tex_status()
    use_pgf = status == "ready" and bool(eng)
    fig_tex = latex_figure(ref + (".pgf" if use_pgf else ".pdf"),
                           caption, slug, pgf=use_pgf,
                           size=spec.get("size", "2col"),
                           env=figure_env(spec), dims=geometry(spec))
    tab = spec.get("table")
    tab_tex = ""
    if tab:
        exp = spec.get("export") or {}
        tab_tex = latex_table(tab["headers"], tab["rows"],
                              table_caption(spec, caption),
                              tab.get("label", slug),
                              cells=exp.get("cells", "ci"),
                              notes=tab.get("notes"), venue=exp.get("venue"))
    w, h, pt = geometry(spec)
    size_txt = f"{w:.2f}×{h:.2f} in, {pt:g} pt"
    if status == "probing":
        hint = (f"Printed at {size_txt} · checking for TeX… (a PDF requested "
                "before the check ends is drawn by Matplotlib)")
    elif use_pgf:
        hint = (f"Printed at {size_txt} · typeset with {eng} in your paper's "
                f"fonts · the LaTeX snippet expects the files at {ref}.*")
    else:
        hint = (f"Printed at {size_txt} · PDF drawn by Matplotlib (install "
                "TeX Live or MiKTeX to typeset figures in your paper's fonts)")
    return fig_tex, tab_tex, hint


def provenance(spec: dict, pdf_method: str) -> dict:
    """What a reader needs to reproduce an exported figure: the KPViz code
    and schema, the catalog it was drawn from (version and a fingerprint of
    every input file's content hash), the export engine, and the settings
    the caption states."""
    import datetime
    import hashlib
    out = {"exported_at": datetime.datetime.now(datetime.timezone.utc)
           .isoformat(timespec="seconds"),
           "pdf_engine": pdf_method}
    try:
        from . import db
        from .scanner import CODE_VERSION
        h = hashlib.blake2b(digest_size=12)
        for rel, fhash, size in db.q("SELECT relpath, hash, size FROM files "
                                     "ORDER BY relpath"):
            h.update(f"{rel}\0{fhash or ''}\0{size}\n".encode())
        out.update(kpviz_code=CODE_VERSION, schema=db.SCHEMA_VERSION,
                   catalog_version=db.scan_version(),
                   inputs_fingerprint=h.hexdigest(),
                   last_scan_ok=bool(db.kv_get("last_scan_ok", True)))
    except Exception as e:                     # never block an export
        out["provenance_error"] = f"{type(e).__name__}: {e}"[:200]
    out["export_options"] = spec.get("export") or {}
    return out


def export_bundle(spec: dict, name: str | None = None) -> bytes:
    """Zip with <stem>.pdf (+ .pgf when TeX typesets it) + .png + .tex
    (+ -table.tex), the figure spec and a provenance manifest. File names
    follow `export_name`, and the .tex references figures/<stem>.* like the
    copied snippet does."""
    import zipfile
    slug, ref = export_name(spec)
    pdf, method, pdf_note = fig_pdf(spec)
    pgf, pgf_err = fig_pgf(spec)
    png = fig_png(spec)
    caption = spec.get("caption", "")
    tex = latex_figure(ref + (".pgf" if pgf else ".pdf"),
                       caption, slugify(spec.get("name", "figure")), pgf=bool(pgf),
                       size=spec.get("size", "2col"), env=figure_env(spec),
                       dims=geometry(spec))
    tab = spec.get("table")
    cells = (spec.get("export") or {}).get("cells") or "ci"
    table_tex = (latex_table(tab["headers"], tab["rows"],
                             table_caption(spec, caption),
                             tab.get("label", slugify(spec.get("name", "figure"))),
                             cells=cells,
                             notes=tab.get("notes"),
                             venue=(spec.get("export") or {}).get("venue"))
                 if tab else None)
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
        z.writestr("provenance.json", json.dumps(provenance(spec, method),
                                                 indent=1, default=str))
        status, eng = tex_status()
        if pgf:
            tex_line = (f"PGF typeset and checked here with {eng}: put {slug}.pgf "
                        f"and {slug}.pdf in your paper's {FIG_DIR}/ folder and "
                        f"paste {slug}.tex (it \\input{{}}s {ref}.pgf; fonts "
                        "follow your document).\n"
                        "Preamble: \\usepackage{pgf} (the .tex already carries "
                        "\\providecommand{\\mathdefault}[1]{#1}, which Matplotlib\n"
                        "writes into every .pgf but never defines).\n")
        else:
            why = pgf_err or ("no TeX distribution found" if not eng
                              else f"{eng} could not typeset it")
            tex_line = (f"No .pgf in this bundle ({why}); {slug}.tex includes "
                        f"{ref}.pdf via \\includegraphics.\n")
        z.writestr("README.txt",
                   f"KPViz export — {name or spec.get('name', slug)}\n"
                   f"PDF rendered via: {method}"
                   + (f" ({pdf_note})" if pdf_note else "") + "\n"
                   + tex_line
                   + "provenance.json records the KPViz code and catalog "
                     "versions and a fingerprint of the input files.\n"
                   + f"Caption:\n{caption}\n")
    return buf.getvalue()
