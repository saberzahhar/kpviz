"""Card layer — the *only* place that interprets the abstract schema.

Keys are contractual, values are data:

  data/documents/document.{dataset}.json      dataset card
  data/documents/document.{dataset}.jsonl     the collection itself
  data/models/model.{model}.json              model card
  data/architectures/architecture.{arch}.json architecture card
  data/insights/scores.jsonl                  cross-document similarity pairs
  data/inferences/{dataset}/{model}/{arch}/{run}/
        run_{run}.json                        run parameters
        batch_%05d.json / .jsonl              batch meta / predictions

Folder tokens link to card file-name tokens. Tokens that match no card
(e.g. an architecture folder called "n.a") stay usable but unresolved:
no assumptions are invented for them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import load_json

UNKNOWN_TOKENS = {"n.a", "na", "n/a", "none", "unknown", "null", "-", ""}


def is_unknown_token(token: str | None) -> bool:
    return (token or "").strip().lower() in UNKNOWN_TOKENS


# --------------------------------------------------------------------------
# Parameter specs (model card "inference" block)
# --------------------------------------------------------------------------
@dataclass
class ParamSpec:
    name: str
    type: str                       # int | float | str | set | context_window | bool
    values: list | None = None      # legal values (str/set)
    min: float | None = None
    max: float | None = None
    default: Any = None
    tokenizer: str | None = None    # context_window params bind a tokenizer
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_card(cls, name: str, spec: dict) -> "ParamSpec":
        return cls(
            name=name,
            type=str(spec.get("type", "str")),
            values=spec.get("values"),
            min=spec.get("min"),
            max=spec.get("max"),
            default=spec.get("default"),
            tokenizer=spec.get("tokenizer"),
            raw=spec,
        )

    @property
    def numeric(self) -> bool:
        return self.type in ("int", "float", "context_window")

    def check(self, value) -> str | None:
        """Return a human-readable violation, or None if the value is legal."""
        t = self.type
        if t in ("int", "context_window"):
            if not isinstance(value, int) or isinstance(value, bool):
                return f"expected int, got {type(value).__name__}"
        elif t == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return f"expected float, got {type(value).__name__}"
        elif t == "bool":
            if not isinstance(value, bool):
                return f"expected bool, got {type(value).__name__}"
        elif t == "str":
            if not isinstance(value, str):
                return f"expected str, got {type(value).__name__}"
            if self.values and value not in self.values:
                return f"value {value!r} not in allowed set"
        elif t == "set":
            if not isinstance(value, (list, tuple)):
                return f"expected list, got {type(value).__name__}"
            if self.values is not None:
                bad = [v for v in value if v not in self.values]
                if bad:
                    return f"illegal members {bad!r}"
        if self.numeric and isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.min is not None and value < self.min:
                return f"{value} < min {self.min}"
            if self.max is not None and value > self.max:
                return f"{value} > max {self.max}"
        return None


# --------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------
@dataclass
class DatasetCard:
    dataset_id: str
    path: Path | None
    raw: dict = field(default_factory=dict)

    @property
    def description(self) -> str:
        return self.raw.get("description", "")

    @property
    def domain(self) -> str | None:
        return self.raw.get("domain")

    @property
    def subdomain(self) -> str | None:
        return self.raw.get("sub-domain")

    @property
    def sections(self) -> dict[str, dict]:
        return self.raw.get("document", {}) or {}

    @property
    def annotations(self) -> dict[str, dict]:
        return self.raw.get("annotations", {}) or {}

    @property
    def metadata_spec(self) -> dict[str, dict]:
        return self.raw.get("metadata", {}) or {}

    def section_langs(self, fieldname: str) -> list[str]:
        return (self.sections.get(fieldname) or {}).get("languages", []) or []

    def ann_langs(self, ann_key: str) -> list[str]:
        return (self.annotations.get(ann_key) or {}).get("languages", []) or []

    @property
    def languages(self) -> list[str]:
        seen: dict[str, None] = {}
        for spec in list(self.sections.values()) + list(self.annotations.values()):
            for l in spec.get("languages", []) or []:
                seen.setdefault(l)
        return list(seen)


@dataclass
class ModelCard:
    model_key: str                       # file-name token == inference folder token
    path: Path | None
    raw: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.raw.get("name") or self.raw.get("model_id") or self.model_key

    @property
    def model_id(self) -> str:
        return self.raw.get("model_id", self.model_key)

    @property
    def family(self) -> list[list[str]]:
        return self.raw.get("family", []) or []

    @property
    def family_top(self) -> str | None:
        f = self.family
        return f[0][0] if f and f[0] else None

    @property
    def backend(self) -> str | None:
        return self.raw.get("backend")

    @property
    def capabilities(self) -> dict:
        return self.raw.get("capabilities", {}) or {}

    @property
    def supervision(self) -> list[str]:
        return self.raw.get("supervision", []) or []

    @property
    def domains(self) -> list[dict]:
        return self.raw.get("domains", []) or []

    @property
    def languages(self) -> list[str]:
        return self.raw.get("languages", []) or []

    @property
    def n_parameters(self) -> int | None:
        return (self.raw.get("parameters") or {}).get("total")

    @property
    def references(self) -> dict:
        return self.raw.get("references", {}) or {}

    @property
    def inference_specs(self) -> dict[str, ParamSpec]:
        return {k: ParamSpec.from_card(k, v)
                for k, v in (self.raw.get("inference") or {}).items()
                if isinstance(v, dict)}

    @property
    def tokenizer_specs(self) -> list[str]:
        out: dict[str, None] = {}
        for ps in self.inference_specs.values():
            if ps.tokenizer:
                out.setdefault(ps.tokenizer)
        return list(out)

    def context_params(self) -> list[ParamSpec]:
        """Parameters that bound the input/output window (type context_window)."""
        return [p for p in self.inference_specs.values()
                if p.type == "context_window"]

    def validate_params(self, params: dict) -> tuple[dict, list[dict]]:
        """Resolve run parameters against this card.

        Returns (resolved, violations). resolved maps every spec'd parameter
        to {'value', 'source': given|default|missing}; violations is a list of
        {'param','value','problem'} — including unknown parameters. Nothing
        raises; illegal values are kept (flagged) so downstream never breaks.
        """
        specs = self.inference_specs
        resolved: dict[str, dict] = {}
        violations: list[dict] = []
        params = params or {}
        for name, spec in specs.items():
            if name in params:
                v = params[name]
                problem = spec.check(v)
                if problem:
                    violations.append({"param": name, "value": v,
                                       "problem": problem})
                resolved[name] = {"value": v, "source": "given"}
            elif spec.default is not None:
                resolved[name] = {"value": spec.default, "source": "default"}
            else:
                resolved[name] = {"value": None, "source": "missing"}
        for name, v in params.items():
            if name not in specs:
                resolved[name] = {"value": v, "source": "given"}
                violations.append({"param": name, "value": v,
                                   "problem": "not declared in model card"})
        return resolved, violations


@dataclass
class ArchCard:
    arch_key: str
    path: Path | None
    raw: dict = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return bool(self.raw)

    @property
    def name(self) -> str:
        return self.raw.get("name") or self.raw.get("arch_id") or self.arch_key

    @property
    def kind(self) -> str | None:
        return self.raw.get("kind")

    @property
    def variables(self) -> dict[str, dict]:
        return self.raw.get("variables", {}) or {}

    @property
    def rates(self) -> dict[str, dict]:
        """{cost_unit: {raw_var: coefficient}} — a linear cost model."""
        return self.raw.get("rates", {}) or {}

    def var_level(self, var: str) -> str | None:
        return (self.variables.get(var) or {}).get("level")

    def needed_vars(self, unit: str) -> list[str]:
        return [v for v, c in (self.rates.get(unit) or {}).items() if c]

    @property
    def cost_units(self) -> list[str]:
        return list(self.rates.keys())


# --------------------------------------------------------------------------
# Tree discovery (pure path logic; no file content here except cards)
# --------------------------------------------------------------------------
_DOC_CARD = re.compile(r"^document\.(?P<ds>.+)\.json$")
_DOC_COLL = re.compile(r"^document\.(?P<ds>.+)\.jsonl$")
_MODEL_CARD = re.compile(r"^model\.(?P<m>.+)\.json$")
_ARCH_CARD = re.compile(r"^architecture\.(?P<a>.+)\.json$")
_RUN_CARD = re.compile(r"^run_(?P<r>.+)\.json$")
_BATCH_META = re.compile(r"^batch_(?P<i>\d+)\.json$")
_BATCH_PREDS = re.compile(r"^batch_(?P<i>\d+)\.jsonl$")


def classify_path(root: Path, path: Path) -> dict | None:
    """Classify one file inside the data tree.

    Returns {'kind', ...link tokens} or None for files we don't own."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    name = path.name
    if len(parts) == 2 and parts[0] == "documents":
        m = _DOC_CARD.match(name)
        if m:
            return {"kind": "dataset_card", "dataset": m["ds"]}
        m = _DOC_COLL.match(name)
        if m:
            return {"kind": "dataset_docs", "dataset": m["ds"]}
    if len(parts) == 2 and parts[0] == "models":
        m = _MODEL_CARD.match(name)
        if m:
            return {"kind": "model_card", "model": m["m"]}
    if len(parts) == 2 and parts[0] == "architectures":
        m = _ARCH_CARD.match(name)
        if m:
            return {"kind": "arch_card", "arch": m["a"]}
    if len(parts) == 2 and parts[0] == "insights" and name.endswith(".jsonl"):
        return {"kind": "scores"}
    if len(parts) == 6 and parts[0] == "inferences":
        _, ds, model, arch, run, fname = parts
        base = {"dataset": ds, "model": model, "arch": arch, "run": run}
        m = _RUN_CARD.match(fname)
        if m:
            return {"kind": "run_card", **base}
        m = _BATCH_META.match(fname)
        if m:
            return {"kind": "batch_meta", "batch_idx": int(m["i"]), **base}
        m = _BATCH_PREDS.match(fname)
        if m:
            return {"kind": "batch_preds", "batch_idx": int(m["i"]), **base}
    return None


def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class CardIndex:
    """All cards, loaded once per scan and shared read-only.

    Folder tokens resolve to cards through *aliases*: the card's file-name
    token first, then its declared id (model_id / arch_id, including the
    last path segment of a URL id) and display name — all normalised. This
    keeps the linkage working whether a run folder says `api`, `openai_api`,
    `bart-base-kp20k` or `bartbasekp20k`."""

    def __init__(self):
        self.datasets: dict[str, DatasetCard] = {}
        self.models: dict[str, ModelCard] = {}
        self.archs: dict[str, ArchCard] = {}
        self._model_alias: dict[str, str] = {}
        self._arch_alias: dict[str, str] = {}

    def _build_aliases(self):
        self._model_alias = {}
        for key, card in self.models.items():
            names = [key, card.name, card.model_id,
                     str(card.model_id).rstrip("/").split("/")[-1]]
            for n in names:
                nk = _norm_token(str(n))
                if nk:
                    self._model_alias.setdefault(nk, key)
        self._arch_alias = {}
        for key, card in self.archs.items():
            names = [key, card.raw.get("arch_id"), card.name]
            for n in names:
                nk = _norm_token(str(n))
                if nk:
                    self._arch_alias.setdefault(nk, key)

    @classmethod
    def load(cls, root: Path) -> "CardIndex":
        idx = cls()
        docs = root / "documents"
        if docs.is_dir():
            for p in sorted(docs.glob("document.*.json")):
                m = _DOC_CARD.match(p.name)
                if m:
                    idx.datasets[m["ds"]] = DatasetCard(m["ds"], p, _safe(p))
            # collections without a card still deserve an entry
            for p in sorted(docs.glob("document.*.jsonl")):
                m = _DOC_COLL.match(p.name)
                if m and m["ds"] not in idx.datasets:
                    idx.datasets[m["ds"]] = DatasetCard(m["ds"], None, {})
        models = root / "models"
        if models.is_dir():
            for p in sorted(models.glob("model.*.json")):
                m = _MODEL_CARD.match(p.name)
                if m:
                    idx.models[m["m"]] = ModelCard(m["m"], p, _safe(p))
        archs = root / "architectures"
        if archs.is_dir():
            for p in sorted(archs.glob("architecture.*.json")):
                m = _ARCH_CARD.match(p.name)
                if m:
                    idx.archs[m["a"]] = ArchCard(m["a"], p, _safe(p))
        idx._build_aliases()
        return idx

    # Folder tokens may or may not resolve; never invent a card.
    def model(self, token: str) -> ModelCard:
        hit = self.models.get(token)
        if hit is None:
            key = self._model_alias.get(_norm_token(token))
            hit = self.models.get(key) if key else None
        return hit or ModelCard(token, None, {})

    def arch(self, token: str) -> ArchCard:
        if is_unknown_token(token):
            return ArchCard(token, None, {})
        hit = self.archs.get(token)
        if hit is None:
            key = self._arch_alias.get(_norm_token(token))
            hit = self.archs.get(key) if key else None
        return hit or ArchCard(token, None, {})

    def dataset(self, token: str) -> DatasetCard:
        return self.datasets.get(token) or DatasetCard(token, None, {})


def _safe(p: Path) -> dict:
    try:
        d = load_json(p)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}
