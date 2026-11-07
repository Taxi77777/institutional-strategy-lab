"""Modele « multi-Fibonacci / nombre d'or » (demande du trader, 7 nov.) : quand les retracements de plusieurs
mouvements (petite, moyenne et grande echelle : sommets / creux sur 3, 8 et 21 bougies, des nombres de Fibonacci)
tombent au meme prix, la zone est surveillee par beaucoup de traders et d'algorithmes : le prix y reagirait.

Regles : sommets et creux confirmes seulement N bougies apres (jamais de donnee future) ; pour chaque echelle, le
dernier mouvement (creux -> sommet = haussier, sommet -> creux = baissier) ; niveaux 61,8 % (« or ») ou 50 / 61,8 /
78,6 % (« multi ») ; confluence = deux niveaux d'echelles differentes a moins de « tol » ATR ; ordre limite sur la
confluence (achat dans un mouvement haussier, vente dans un baissier), valable 24 bougies, annule si l'objectif est
atteint avant ; stop derriere l'origine du mouvement (ou 1 ATR) ; objectif : retour au sommet (creux) du mouvement
(>= 1 fois le risque) ; moitie au premier objectif puis stop a l'entree. Regle stricte : l'objectif n'est jamais
compte sur la bougie du remplissage. Test honnete : 60 % anciens pour choisir, 40 % recents pour le verdict.
   python -m lab.fibo fx 1H"""
import json
import logging
import os
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from lab.liq2liq import _atr
from lab.ob_retour import play

log = logging.getLogger("lab")
SCALES = (3, 8, 21)
RATIOS = {"or": (0.618,), "multi": (0.5, 0.618, 0.786)}


def pivots(h, l, N):
    """Liste alternee (bougie de confirmation, type +1 sommet / -1 creux, prix)."""
    n = len(h)
    if n < 2 * N + 2:
        return []
    mx = swv(h, 2 * N + 1).max(axis=1)
    mn = swv(l, 2 * N + 1).min(axis=1)
    ev = []
    for k in range(len(mx)):
        i = k + N
        if h[i] == mx[k]:
            ev.append((i, 1, h[i]))
        if l[i] == mn[k]:
            ev.append((i, -1, l[i]))
    out = []
    for i, t, p in ev:
        if out and out[-1][1] == t:
            if (p > out[-1][2]) if t == 1 else (p < out[-1][2]):
                out[-1] = (i + N, t, p)
            continue
        out.append((i + N, t, p))
    return out


def setups(a, atr, piv, ratios="or", tol=0.1, stop="origine", kz=True, wait=24):
    ts, h, l, c = a[:, 0], a[:, 2], a[:, 3], a[:, 4]
    hr = ((ts % 86400) // 3600).astype(int)
    n = len(a)
    conf = sorted({p[0] for pv in piv.values() for p in pv if p[0] < n - 1})
    ptr = {s: 0 for s in piv}
    out, seen = [], set()
    for t in conf:
        legs = {}
        for s, pv in piv.items():
            while ptr[s] + 1 < len(pv) and pv[ptr[s] + 1][0] <= t:
                ptr[s] += 1
            k = ptr[s]
            if k >= 1 and pv[k][0] <= t:
                last, prev = pv[k], pv[k - 1]
                legs[s] = (last[1], prev[2], last[2])      # (sens du mouvement, origine, extremite)
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        for d in (1, -1):
            L = [(s, o, e) for s, (dd, o, e) in legs.items() if dd == d and (e - o) * d > 0]
            if len(L) < 2:
                continue
            lv = [(s, e - d * r * (e - o), o, e) for s, o, e in L for r in RATIOS[ratios]]
            best = None
            for x in range(len(lv)):
                for y in range(x + 1, len(lv)):
                    if lv[x][0] == lv[y][0] or abs(lv[x][1] - lv[y][1]) > tol * atr[t]:
                        continue
                    e = (lv[x][1] + lv[y][1]) / 2
                    if (c[t] - e) * d < 0.2 * atr[t]:
                        continue                          # le prix doit encore revenir sur la zone
                    if best is None or abs(c[t] - e) < abs(c[t] - best[0]):
                        best = (e, lv[x], lv[y])
            if best is None:
                continue
            e, p1, p2 = best
            key = (d,) + tuple(sorted((p1[2], p1[3], p2[2], p2[3])))   # meme paire de mouvements
            if key in seen:
                continue
            seen.add(key)
            orig = min(p1[2], p2[2]) if d == 1 else max(p1[2], p2[2])
            sl = orig - d * 0.1 * atr[t] if stop == "origine" else e - d * atr[t]
            risk = (e - sl) * d
            tg = max(p1[3], p2[3]) if d == 1 else min(p1[3], p2[3])
            if risk < 0.2 * atr[t] or (tg - e) * d < risk:
                continue
            for m in range(t + 1, min(t + 1 + wait, n)):
                if (l[m] <= e) if d == 1 else (h[m] >= e):
                    if not kz or 7 <= hr[m] < 17:
                        out.append({"i": m, "dir": d, "entry": float(e), "stop": float(sl), "target": float(tg)})
                    break
                if (h[m] >= tg) if d == 1 else (l[m] <= tg):
                    break
    out.sort(key=lambda x: x["i"])
    return out


VARIANTES = [{"ratios": r, "tol": t, "stop": s, "tp1": p} for r in ("or", "multi") for t in (0.1, 0.25)
             for s in ("origine", "atr") for p in (0.5, 1.0)]


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, tf)
            if len(a) > 1500:
                bars[s] = (a, _atr(a), {N: pivots(a[:, 2], a[:, 3], N) for N in SCALES})
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([b[0][:, 0] for b in bars.values()])
    cut = np.quantile(t_all, 0.6)
    dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
    cache, res = {}, []
    for v in VARIANTES:
        tr, te = [], []
        for s, (a, atr, piv) in bars.items():
            k = (s, v["ratios"], v["tol"], v["stop"])
            if k not in cache:
                cache[k] = setups(a, atr, piv, v["ratios"], v["tol"], v["stop"], kz=(kind == "fx"))
            busy = -1
            for st in cache[k]:
                if st["i"] <= busy:
                    continue
                p = play(a, st, kind, tp1_r=v["tp1"])
                if p is None:
                    continue
                r, win, end = p
                busy = end
                (te if a[st["i"], 0] >= cut else tr).append((r, win))

        def stats(x, days):
            if not x:
                return {"trades": 0}
            r = np.array([y[0] for y in x])
            return {"trades": len(x), "reussite": round(float(np.mean([y[1] for y in x])), 3),
                    "gain_moyen_R": round(float(r.mean()), 3), "total_R": round(float(r.sum()), 1),
                    "par_jour": round(len(x) / max(1, days), 2)}
        res.append({"variante": v, "apprentissage": stats(tr, dtr), "test": stats(te, dte)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    best = max(res, key=lambda x: x["apprentissage"].get("gain_moyen_R", -9)
               if x["apprentissage"].get("trades", 0) >= 50 else -9)
    print("CHOISIE (sur l'apprentissage seulement)", json.dumps(best["variante"]), "-> TEST (jamais vu)",
          json.dumps(best["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/fibo_%s_%s.json" % (kind, tf), "w") as f:
        json.dump({"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "actifs": len(bars), "choisie": best["variante"], "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
