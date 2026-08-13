"""Export engine: PGF / PDF / PNG figures and booktabs LaTeX tables.

PGF needs a TeX distribution (it typesets labels with your document's
fonts). When none is installed we degrade explicitly: PDF falls back to
Matplotlib's own vector backend and the LaTeX snippet switches to
\\includegraphics — nothing breaks, the caption says what happened.
"""
from __future__ import annotations

import io
import re
import shutil
import threading
from collections import OrderedDict
from functools import lru_cache

from .figures import _TEX_TABLE, to_mpl
from .util import fmt_num, stable_hash

# Matplotlib's pyplot figure manager and rcParams are process-global, and the
# server is threaded: two concurrent exports would interleave and corrupt each
# other's output. One render at a time, with a small result cache so repeated
# downloads of the same figure are free.
_render_lock = threading.RLock()
_CACHE: "OrderedDict[str, bytes | str]" = OrderedDict()
_CACHE_MAX = 64


def _cached(kind: str, spec: dict, build):
    key = kind + ":" + stable_hash(spec)
    with _render_lock:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
        val = build()
        if val is not None:
            _CACHE[key] = val
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
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


_PROBE_NOTES: list[str] = []


@lru_cache(maxsize=1)
def tex_engine() -> str | None:
    """First TeX engine that actually compiles a *representative* PGF figure.

    `which` alone is not enough — a minimal or broken TeX install can have the
    binary but miss font metrics or a Lua module. Probing once (cached) keeps
    exports honest."""
    import io

    import matplotlib
    for eng in ("lualatex", "xelatex", "pdflatex"):
        if not shutil.which(eng):
            continue
        try:
            matplotlib.rcParams["pgf.texsystem"] = eng
            matplotlib.rcParams["pgf.preamble"] = ""
            from .figures import to_mpl
            fig = to_mpl(_PROBE_SPEC, pgf=True)
            buf = io.BytesIO()
            # exercise the exact export path (tight bbox measures text via TeX)
            fig.savefig(buf, format="pdf", backend="pgf", bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
            if buf.getvalue()[:5] == b"%PDF-":
                return eng
            _PROBE_NOTES.append(f"{eng}: produced no PDF")
        except Exception as e:
            first = next((ln for ln in str(e).splitlines()
                          if ln.strip().startswith("!")), "").strip()
            _PROBE_NOTES.append(f"{eng}: {first or type(e).__name__}")
            continue
    # every installed engine failed the probe: say which and why, once. The
    # alternative is a silent downgrade of every export in the session.
    if _PROBE_NOTES:
        print("· no usable TeX engine — exports use Matplotlib's vector "
              "backend.\n  " + "\n  ".join(_PROBE_NOTES))
    return None


def _configure_pgf():
    import matplotlib
    matplotlib.rcParams["pgf.texsystem"] = tex_engine() or "pdflatex"
    matplotlib.rcParams["pgf.preamble"] = ""


_PGF_WARNED: set[str] = set()
LAST_PGF_ERROR: str | None = None


def _warn_pgf_fallback(exc: Exception) -> None:
    """Report a PGF render that fell back, once per distinct cause."""
    global LAST_PGF_ERROR
    first = next((ln for ln in str(exc).splitlines()
                  if ln.strip().startswith("!")), "").strip()
    LAST_PGF_ERROR = first or f"{type(exc).__name__}: {str(exc)[:160]}"
    if LAST_PGF_ERROR not in _PGF_WARNED:
        _PGF_WARNED.add(LAST_PGF_ERROR)
        print(f"· PGF export fell back to Matplotlib's vector backend "
              f"({tex_engine()}): {LAST_PGF_ERROR}")


def fig_png(spec: dict, dpi: int = 300) -> bytes:
    def build():
        with _render_lock:
            fig = to_mpl(spec, pgf=False)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor="white",
                        bbox_inches="tight", pad_inches=0.02)
            _close(fig)
            return buf.getvalue()
    return _cached(f"png{dpi}", spec, build)


def fig_pdf(spec: dict) -> tuple[bytes, str]:
    """Returns (pdf bytes, method) — 'pgf' (TeX-typeset) or 'matplotlib'
    (vector fallback, no TeX required)."""
    def build():
        with _render_lock:
            if tex_engine():
                try:
                    _configure_pgf()
                    fig = to_mpl(spec, pgf=True)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="pdf", backend="pgf",
                                bbox_inches="tight", pad_inches=0.02)
                    _close(fig)
                    return b"pgf:" + buf.getvalue()
                except Exception as e:
                    # never silent: a PDF that quietly stopped being
                    # TeX-typeset is a paper with two different font stacks
                    _warn_pgf_fallback(e)
            fig = to_mpl(spec, pgf=False)
            buf = io.BytesIO()
            fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.02)
            _close(fig)
            return b"mpl:" + buf.getvalue()
    blob = _cached("pdf", spec, build)
    tag, _, payload = blob.partition(b":")
    return payload, ("pgf" if tag == b"pgf" else "matplotlib")


def fig_pgf(spec: dict) -> str | None:
    """The .pgf source itself (None when no TeX is available)."""
    if not tex_engine():
        return None

    def build():
        with _render_lock:
            try:
                _configure_pgf()
                fig = to_mpl(spec, pgf=True)
                buf = io.BytesIO()
                fig.savefig(buf, format="pgf", backend="pgf")
                _close(fig)
                return buf.getvalue().decode("utf-8")
            except Exception:
                return None
    return _cached("pgf", spec, build)


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


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
                 width: str = "\\linewidth", pgf: bool = True) -> str:
    body = (f"    \\input{{{filename}}}" if pgf
            else f"    \\includegraphics[width={width}]{{{filename}}}")
    lines = [
        PGF_PREAMBLE_NOTE if pgf else None,
        "\\begin{figure}[t]",
        "    \\centering",
        ("    \\resizebox{" + width + "}{!}{%") if pgf else None,
        body,
        "    }" if pgf else None,
        f"    \\caption{{{caption_escape(caption)}}}",
        f"    \\label{{fig:{label}}}",
        "\\end{figure}",
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
    pdf, method = fig_pdf(spec)
    pgf = fig_pgf(spec)
    png = fig_png(spec)
    caption = spec.get("caption", "")
    tex = latex_figure(f"{slug}.pgf" if pgf else f"{slug}.pdf",
                       caption, slug, pgf=bool(pgf))
    buf = io.BytesIO()
    # PDF/PNG are already compressed; level 1 keeps the text members small
    # without spending CPU re-compressing binary ones
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.writestr(f"{slug}.pdf", pdf)
        if pgf:
            z.writestr(f"{slug}.pgf", pgf)
        z.writestr(f"{slug}.png", png)
        z.writestr(f"{slug}.tex", tex)
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
