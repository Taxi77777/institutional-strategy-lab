"""L'Inventeur (21e agent) : invente de nouveaux indicateurs en combinant ceux des 20 specialistes, les fait tester par
le labo avec les memes regles, garde en memoire ceux qui ont donne des strategies validees et les fait evoluer
(mutation, croisement) d'une recherche a l'autre. Gratuit, sans service externe."""
import json
import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from .indicators import safe_div, shift, sma

OPS = ("accord", "produit", "ecart_rang", "acceleration", "croise_moyenne", "ratio", "divergence_k")
MEMORY = "ideas.json"


def _rank(x, p=100):
    """Rang glissant (0..1) : rend deux indicateurs d'unites differentes comparables, sans regarder le futur."""
    out = np.full(len(x), np.nan)
    if len(x) >= p:
        w = swv(x, p)
        cur = w[:, -1:]
        with np.errstate(invalid="ignore"):
            out[p - 1:] = np.where(np.isfinite(cur[:, 0]), (w <= cur).sum(axis=1) / np.maximum(np.isfinite(w).sum(axis=1), 1), np.nan)
    return out


def apply_spec(F, sp):
    a = F.get(sp["a"][0], {}).get(sp["a"][1])
    b = F.get(sp["b"][0], {}).get(sp["b"][1]) if sp.get("b") else None
    if a is None or (sp.get("b") and b is None):
        return None
    k, op = int(sp.get("k", 3)), sp["op"]
    if op == "accord":
        return np.sign(a) * np.sign(b)
    if op == "produit":
        return a * b
    if op == "ecart_rang":
        return _rank(a) - _rank(b)
    if op == "acceleration":
        d = a - shift(a, k)
        return d - shift(d, k)
    if op == "croise_moyenne":
        return a - sma(np.nan_to_num(a), max(3, k * 3))
    if op == "ratio":
        return np.clip(safe_div(a, np.abs(b) + 1e-9), -50, 50)
    if op == "divergence_k":
        return np.sign(a - shift(a, k)) - np.sign(b - shift(b, k))
    return None


def name(sp):
    b = "%s.%s" % tuple(sp["b"]) if sp.get("b") else "-"
    return "%s(%s.%s,%s,%d)" % (sp["op"], sp["a"][0], sp["a"][1], b, int(sp.get("k", 3)))


def load_memory():
    try:
        with open(MEMORY) as f:
            m = json.load(f)
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def load_scout(path="veille.json"):
    """Poids par specialiste publies par l'Eclaireur (veille GitHub / arXiv) ; vide si absent."""
    try:
        with open(path) as f:
            return json.load(f).get("poids", {}) or {}
    except Exception:
        return {}


def generate(feature_names, n=150, seed=0, memory=None, weights=None):
    """Idees de la semaine : les gagnantes passees, leurs mutations / croisements, puis des idees nouvelles,
       tirees plus souvent dans les familles que l'Eclaireur voit monter (weights)."""
    rng = random.Random(seed)
    feats = [tuple(f) for f in feature_names if f[0] not in ("temps",)]
    w = [float((weights or {}).get(f[0], 1.0)) for f in feats]
    mem = sorted((memory or {}).get("gagnantes", []), key=lambda x: -x.get("score", 0))[:40]
    specs = []
    for g in mem:
        sp = g["spec"]
        specs.append(sp)
        mut = dict(sp, k=max(1, int(sp.get("k", 3)) + rng.choice((-2, -1, 1, 2))))
        specs.append(mut)
        if feats:
            specs.append(dict(sp, b=list(rng.choice(feats)), op=rng.choice(OPS)))
    while len(specs) < n and feats:
        a, b = rng.choices(feats, weights=w, k=2) if len(feats) > 1 else (feats[0], feats[0])
        specs.append({"op": rng.choice(OPS), "a": list(a), "b": list(b), "k": rng.choice((1, 2, 3, 5, 8))})
    uniq = {}
    for sp in specs:
        uniq.setdefault(name(sp), sp)
    return list(uniq.values())[:max(n, len(mem) * 3)]


def remember(final, specs_by_name, memory):
    """Ajoute a la memoire les idees presentes dans les strategies retenues (score = meilleur taux de reussite)."""
    mem = {g["nom"]: g for g in (memory or {}).get("gagnantes", [])}
    for s in final:
        for ag, nm, _lo, _hi in s["conds"]:
            if ag == "inventeur" and nm in specs_by_name:
                sc = s["stats"]["test"]["win"]
                if nm not in mem or sc > mem[nm]["score"]:
                    mem[nm] = {"nom": nm, "spec": specs_by_name[nm], "score": sc}
    return {"gagnantes": sorted(mem.values(), key=lambda g: -g["score"])[:200]}
