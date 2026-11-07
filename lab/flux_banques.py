"""Deux tests demandes par le trader (7 nov.) apres la recherche sur les methodes des banques et des fonds :

A. FLUX DU FIXING DE LONDRES (16 h heure de Londres, prix de reference WM/Reuters) - forex.
   Les gerants (fonds de pension, BlackRock...) executent leurs ordres au fixing ; le dernier jour ouvre du mois, ils
   reequilibrent leurs couvertures de change (gros flux de dollars, a date et heure connues).
   - fin_mois_avant : quelques heures avant le fixing du dernier jour du mois, contre (ou avec) le mouvement du mois,
     sortie au fixing ;
   - fin_mois_apres / jour_apres : juste apres le fixing, contre le mouvement des 3 heures d'avant (le prix retombe
     quand le flux est passe : Evans 2018), sortie 1 a 4 h plus tard ; le dernier jour du mois ou tous les jours ;
   - jour_avant : la tendance de la matinee continue jusqu'au fixing (les banques executent par tranches).

B. CONTRE LA FOULE (les petits traders perdent majoritairement) :
   - forex : position nette des petits traders (« non declarables ») du rapport COT de la CFTC, publie chaque
     semaine ; quand elle est extreme (percentile sur 3 ans) on prend le sens inverse pendant 1 a 2 semaines ;
   - crypto : taux de financement des perpetuels sur 3 jours ; quand la foule paie tres cher pour etre acheteuse
     (vendeuse), on vend (achete) pendant 1 a 3 jours.

Methode honnete habituelle : 60 % les plus anciens pour choisir la variante, 40 % recents (jamais vus) pour le
verdict ; frais deduits ; un trade a la fois par actif.   python -m lab.flux_banques fx 1H"""
import datetime
import json
import logging
import os
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

from lab.liq2liq import COST

log = logging.getLogger("lab")
LDN = ZoneInfo("Europe/London")


def _price(pos, a, t):
    """Prix a l'instant t : ouverture de la bougie qui commence a t, sinon cloture de celle qui finit a t."""
    i = pos.get(int(t))
    if i is not None:
        return a[i, 1]
    i = pos.get(int(t) - 3600)
    return a[i, 4] if i is not None else None


def fixing_trades(a, v):
    ts = a[:, 0]
    pos = {int(t): i for i, t in enumerate(ts)}
    days = sorted({datetime.datetime.fromtimestamp(t, LDN).date() for t in ts})
    days = [d for d in days if d.weekday() < 5]
    last_of_month = {}
    for d in days:
        last_of_month[(d.year, d.month)] = d
    month_open = {}
    for i, t in enumerate(ts):
        d = datetime.datetime.fromtimestamp(t, LDN)
        month_open.setdefault((d.year, d.month), a[i, 1])
    out = []
    for d in days:
        is_me = last_of_month.get((d.year, d.month)) == d
        if v["m"].startswith("fin_mois") and not is_me:
            continue
        tf = datetime.datetime(d.year, d.month, d.day, 16, 0, tzinfo=LDN).timestamp()
        if v["m"] == "fin_mois_avant":
            te, tx = tf - v["h"] * 3600, tf
            pe, p0 = _price(pos, a, te), month_open.get((d.year, d.month))
            if pe is None or p0 is None:
                continue
            s = -np.sign(pe / p0 - 1)
        elif v["m"] == "jour_avant":
            t7 = datetime.datetime(d.year, d.month, d.day, 8, 0, tzinfo=LDN).timestamp()
            te, tx = tf - 3600, tf
            pe, p0 = _price(pos, a, te), _price(pos, a, t7)
            if pe is None or p0 is None:
                continue
            s = -np.sign(pe / p0 - 1)
        else:                                               # apres le fixing : contre les 3 h d'avant
            te, tx = tf, tf + v["h"] * 3600
            pe, p0 = _price(pos, a, te), _price(pos, a, tf - 3 * 3600)
            if pe is None or p0 is None:
                continue
            s = -np.sign(pe / p0 - 1)
        if v["sens"] == "avec":
            s = -s
        px = _price(pos, a, tx)
        if px is None or s == 0:
            continue
        out.append((te, float(s * (px / pe - 1) - COST["fx"])))
    return out


def cot_trades(a, cot, sym, v):
    """a : bougies jour. Signal le lundi qui suit chaque rapport (disponibilite prudente deja dans data.cot)."""
    b, q = cot.get(sym[:3]), cot.get(sym[3:])
    ref = b if b is not None and b.shape[1] > 4 else q
    if ref is None or ref.shape[1] <= 4:
        return []
    tw = ref[:, 0]

    def at(x):
        if x is None or x.shape[1] <= 4:
            return np.zeros(len(tw))
        idx = np.searchsorted(x[:, 0], tw, side="right") - 1
        return np.where(idx >= 0, x[np.clip(idx, 0, None), 4], np.nan)
    x = at(b) - at(q)                                        # la foule est acheteuse de la devise de base
    ts, o = a[:, 0], a[:, 1]
    out, busy = [], -np.inf
    for k in range(60, len(tw)):
        past = x[max(0, k - 156):k]
        past = past[np.isfinite(past)]
        if len(past) < 52 or not np.isfinite(x[k]) or tw[k] < busy:
            continue
        p = np.mean(past < x[k])
        s = -1 if p >= v["q"] else (1 if p <= 1 - v["q"] else 0)
        if not s:
            continue
        if v["sens"] == "avec":
            s = -s
        i = np.searchsorted(ts, tw[k])
        j = np.searchsorted(ts, tw[k] + v["jours"] * 86400)
        if i >= len(ts) or j >= len(ts):
            continue
        out.append((ts[i], float(s * (o[j] / o[i] - 1) - COST["fx"])))
        busy = ts[j]
    return out


def funding_trades(a, f, v):
    """a : bougies jour (ouverture 0 h UTC) ; f : (n, 2) temps, taux horaire relatif."""
    ts, o = a[:, 0], a[:, 1]
    cs = np.r_[0, np.cumsum(f[:, 1])]
    fd = np.full(len(ts), np.nan)
    for i, t in enumerate(ts):                               # moyenne des 72 h connues avant l'ouverture
        lo, hi = np.searchsorted(f[:, 0], t - 72 * 3600), np.searchsorted(f[:, 0], t)
        if hi - lo >= 24:
            fd[i] = (cs[hi] - cs[lo]) / (hi - lo)
    out, busy = [], -1
    for i in range(90, len(ts) - v["jours"]):
        if i < busy or not np.isfinite(fd[i]):
            continue
        past = fd[i - 90:i]
        past = past[np.isfinite(past)]
        if len(past) < 60:
            continue
        p = np.mean(past < fd[i])
        s = -1 if p >= v["q"] else (1 if p <= 1 - v["q"] else 0)
        if not s:
            continue
        if v["sens"] == "avec":
            s = -s
        j = i + v["jours"]
        out.append((ts[i], float(s * (o[j] / o[i] - 1) - COST["cr"])))
        busy = j
    return out


VAR_FIX = ([{"m": "fin_mois_avant", "h": h, "sens": s} for h in (3, 6) for s in ("contre", "avec")] +
           [{"m": m, "h": h, "sens": s} for m in ("fin_mois_apres", "jour_apres") for h in (1, 2, 4)
            for s in ("contre", "avec")] +
           [{"m": "jour_avant", "sens": s} for s in ("contre", "avec")])
VAR_FOULE = [{"q": q, "jours": j, "sens": s} for q in (0.8, 0.9) for j in (5, 10) for s in ("contre", "avec")]
VAR_FUND = [{"q": q, "jours": j, "sens": s} for q in (0.8, 0.9) for j in (1, 3) for s in ("contre", "avec")]


def _stats(x, days):
    if not x:
        return {"trades": 0}
    r = np.array(x)
    return {"trades": len(r), "reussite": round(float(np.mean(r > 0)), 3),
            "gain_moyen_pct": round(100 * float(r.mean()), 4), "total_pct": round(100 * float(r.sum()), 2),
            "par_jour": round(len(r) / max(1, days), 3)}


def _evaluate(name, variants, trades_of, t_all):
    cut = np.quantile(t_all, 0.6)
    dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
    res = []
    for v in variants:
        tr, te = [], []
        for t, r in trades_of(v):
            (te if t >= cut else tr).append(r)
        res.append({"modele": name, "variante": v, "apprentissage": _stats(tr, dtr), "test": _stats(te, dte)})
        print("VARIANTE", name, json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    best = max(res, key=lambda x: x["apprentissage"].get("gain_moyen_pct", -9)
               if x["apprentissage"].get("trades", 0) >= 30 else -9)
    print("CHOISIE", name, "(sur l'apprentissage seulement)", json.dumps(best["variante"]), "-> TEST (jamais vu)",
          json.dumps(best["test"]), flush=True)
    return {"choisie": best["variante"], "variantes": res}


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    out = {}
    if kind == "fx":
        h1 = {}
        for s in UNIVERSE["fx"]:
            try:
                a = data.long_bars(s, "fx", "1H")
                if len(a) > 2000:
                    h1[s] = a
            except Exception as ex:
                log.warning("%s : %s", s, ex)
        if h1:
            t_all = np.concatenate([a[:, 0] for a in h1.values()])
            out["fixing"] = _evaluate("fixing", VAR_FIX,
                                      lambda v: [x for a in h1.values() for x in fixing_trades(a, v)], t_all)
        try:
            cot = data.cot(years=11)
        except Exception as ex:
            log.warning("COT : %s", ex)
            cot = {}
        d1 = {}
        for s in UNIVERSE["fx"]:
            try:
                d1[s] = data.long_bars(s, "fx", "1D")
            except Exception as ex:
                log.warning("%s 1D : %s", s, ex)
        d1 = {s: a for s, a in d1.items() if len(a) > 300}
        if cot and d1:
            t_all = np.concatenate([a[:, 0] for a in d1.values()])
            out["foule_cot"] = _evaluate("foule_cot", VAR_FOULE,
                                         lambda v: [x for s, a in d1.items() for x in cot_trades(a, cot, s, v)], t_all)
    else:
        bars = {}
        for s in UNIVERSE["cr"]:
            try:
                a = data.long_bars(s, "cr", "1D")
                f = data.funding(s)
                if len(a) > 200 and len(f) > 500:
                    bars[s] = (a, f)
            except Exception as ex:
                log.warning("%s : %s", s, ex)
        if bars:
            t_all = np.concatenate([a[:, 0] for a, f in bars.values()])
            out["foule_financement"] = _evaluate("foule_financement", VAR_FUND,
                                                 lambda v: [x for a, f in bars.values() for x in funding_trades(a, f, v)],
                                                 t_all)
            out["actifs"] = len(bars)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/flux_banques_%s.json" % kind, "w") as fh:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), **out}, fh, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
