"""Strategie « suivi de tendance » (11 nov., proposee par l'assistant) : bougies jour, cassure du plus haut / plus bas
des N derniers jours dans le sens de la tendance de fond (cloture vs cloture d'il y a 90 jours), entree a l'ouverture
suivante, stop a k ATR qui suit la meilleure cloture, aucun objectif (on laisse courir). Une position par actif.
Resultat en multiples du risque (R), frais deduits ; risque identique par trade. Choix de la variante sur les 60 %
de trades les plus anciens, verdict sur les 40 % recents.   python -m lab.tendance cr"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import COST

log = logging.getLogger("lab")
VARIANTES = [{"n": n, "fond": f, "k": k} for n in (20, 55) for f in (0, 90) for k in (2.0, 3.0)]


def _atr(a, n=20):
    h, l, c = a[:, 2], a[:, 3], a[:, 4]
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    out = np.full(len(a), np.nan)
    cs = np.cumsum(tr)
    out[n:] = (cs[n:] - cs[:-n]) / n
    return out


def trades(a, kind, n=20, fond=90, k=2.0):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    atr = _atr(a)
    out, i, N = [], max(n, fond, 21) + 1, len(a)
    while i < N - 2:
        d = 0
        if np.isfinite(atr[i]) and atr[i] > 0:
            if c[i] > h[i - n:i].max() and (not fond or c[i] > c[i - fond]):
                d = 1
            elif c[i] < l[i - n:i].min() and (not fond or c[i] < c[i - fond]):
                d = -1
        if not d:
            i += 1
            continue
        e, risk = o[i + 1], k * atr[i]
        stop, best, j, x = e - d * risk, e, i + 1, None
        while j < N:
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x = min(o[j], stop) if d == 1 else max(o[j], stop)
                if j == i + 1:
                    x = stop
                break
            best = max(best, c[j]) if d == 1 else min(best, c[j])
            stop = max(stop, best - risk) if d == 1 else min(stop, best + risk)
            j += 1
        if x is None:
            break                                            # trade encore ouvert a la fin : non compte
        out.append((ts[i + 1], ts[j], d, float((x - e) * d / risk - COST[kind] * e / risk)))
        i = j + 1
    return out


def run(kind, tf="1D"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, "1D")
            if kind == "fx":
                a = a[((a[:, 0] // 86400 + 3) % 7) != 6]
            if len(a) > 400:
                bars[s] = a
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = np.quantile(t_all, 0.6)
    print("ACTIFS", kind, len(bars), "du", time.strftime("%d/%m/%Y", time.gmtime(t_all.min())), "coupure",
          time.strftime("%d/%m/%Y", time.gmtime(cut)), "au", time.strftime("%d/%m/%Y", time.gmtime(t_all.max())), flush=True)

    def stats(x):
        if not x:
            return {"trades": 0}
        x = sorted(x, key=lambda t: t[1])
        r = np.array([t[3] for t in x])
        eq = np.cumsum(r)
        dd = float((np.maximum.accumulate(eq) - eq).max())
        yrs = max(0.5, (x[-1][1] - x[0][0]) / 86400 / 365)
        w, l_ = r[r > 0], r[r <= 0]
        return {"trades": len(r), "reussite": round(float((r > 0).mean()), 3), "gain_moyen_R": round(float(r.mean()), 3),
                "gain_moyen_gagnant": round(float(w.mean()), 2) if len(w) else 0,
                "perte_moyenne": round(float(l_.mean()), 2) if len(l_) else 0, "total_R": round(float(r.sum()), 1),
                "R_par_an": round(float(r.sum()) / yrs, 1), "pire_baisse_R": round(dd, 1),
                "achats_R": round(float(sum(t[3] for t in x if t[2] == 1)), 1),
                "ventes_R": round(float(sum(t[3] for t in x if t[2] == -1)), 1)}
    res = []
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            for t in trades(a, kind, **v):
                (te if t[0] >= cut else tr).append(t)
        res.append({"variante": v, "apprentissage": stats(tr), "test": stats(te)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    best = max(res, key=lambda x: x["apprentissage"].get("gain_moyen_R", -9) if x["apprentissage"].get("trades", 0) >= 100 else -9)
    print("CHOISIE (sur l'apprentissage seulement)", json.dumps(best["variante"]), "-> TEST (jamais vu)", json.dumps(best["test"]),
          flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/tendance_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "actifs": len(bars),
                   "choisie": best["variante"], "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(*sys.argv[1:])
