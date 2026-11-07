"""Modele « retour sur l'order block » (demande du trader, 7 nov.) : quand les banques poussent fort le prix (bougie
d'impulsion qui casse la structure), une partie de leurs ordres reste non executee dans l'order block d'origine (la
derniere bougie inverse avant l'impulsion). Au PREMIER retour du prix dans cette zone, elles completent ou ferment
leurs positions : le prix repart dans le sens de l'impulsion.

Regles : impulsion >= k ATR qui cloture au-dela du plus haut (plus bas) des 10 bougies precedentes ; order block =
derniere bougie inverse dans les 3 bougies d'avant ; entree au premier retour (bord ou milieu de la zone) dans les
96 bougies ; stop derriere la zone (+0,1 ATR) ; objectif : le sommet de l'impulsion ou 2 fois le risque ; moitie a
1 fois le risque puis stop a l'entree ; sortie forcee apres 64 bougies. Test honnete : 60 % anciens pour choisir,
40 % recents (jamais vus) pour le verdict.   python -m lab.ob_retour fx 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import COST, _atr

log = logging.getLogger("lab")


def setups(a, atr, k=1.5, entry="bord", cible="impulsion", kz=True):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((ts % 86400) // 3600).astype(int)
    n, out = len(a), []
    for j in range(12, n - 2):
        if not np.isfinite(atr[j]) or atr[j] <= 0:
            continue
        body = c[j] - o[j]
        for d in (1, -1):
            if d * body < k * atr[j]:
                continue
            if not ((c[j] > h[j - 10:j].max()) if d == 1 else (c[j] < l[j - 10:j].min())):
                continue                                         # pas de cassure de structure
            ob = next((m for m in range(j - 1, j - 4, -1) if d * (c[m] - o[m]) < 0), None)
            if ob is None:
                continue
            zlo, zhi = l[ob], h[ob]
            e = (zhi if d == 1 else zlo) if entry == "bord" else (zlo + zhi) / 2
            sl = zlo - 0.1 * atr[j] if d == 1 else zhi + 0.1 * atr[j]
            risk = (e - sl) * d
            if risk < 0.2 * atr[j]:
                continue
            ext = h[j] if d == 1 else l[j]
            for m in range(j + 1, min(j + 97, n)):               # premier retour dans la zone
                if (l[m] <= e) if d == 1 else (h[m] >= e):
                    if m < j + 3 or (kz and not 7 <= hr[m] < 17):
                        break
                    tg = ext if cible == "impulsion" else e + d * 2 * risk
                    if (tg - e) * d < risk:
                        break
                    out.append({"i": m, "dir": d, "entry": float(e), "stop": float(sl), "target": float(tg)})
                    break
                ext = max(ext, h[m]) if d == 1 else min(ext, l[m])
    return out


def play(a, s, kind, max_bars=64):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    m, d, e, sl, tg = s["i"], s["dir"], s["entry"], s["stop"], s["target"]
    if (o[m] - e) * d < 0:
        e = o[m]                                                # ouverture deja au-dela : rempli a l'ouverture
    risk = (e - sl) * d
    if risk <= 0:
        return None
    tp1, half, stop = e + d * risk, False, sl
    cost = COST[kind] * e / risk
    for k in range(m, min(m + max_bars, len(a))):
        if (l[k] <= stop) if d == 1 else (h[k] >= stop):
            return (0.5 + 0.5 * (stop - e) * d / risk if half else (stop - e) * d / risk) - cost, half, k
        if not half and ((h[k] >= tp1) if d == 1 else (l[k] <= tp1)):
            half, stop = True, e
        if (h[k] >= tg) if d == 1 else (l[k] <= tg):
            return (0.5 + 0.5 * (tg - e) * d / risk) - cost, True, k
    k = min(m + max_bars, len(a)) - 1
    r = (0.5 + 0.5 * (c[k] - e) * d / risk) if half else (c[k] - e) * d / risk
    return r - cost, half, k


VARIANTES = [{"k": k, "entry": e, "cible": t} for k in (1.0, 1.5, 2.0) for e in ("bord", "milieu")
             for t in ("impulsion", "2R")]


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, tf)
            if len(a) > 1500:
                bars[s] = a
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([b[:, 0] for b in bars.values()])
    cut = np.quantile(t_all, 0.6)
    dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
    res = []
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            atr = _atr(a)
            busy = -1
            for st in setups(a, atr, kz=(kind == "fx"), **v):
                if st["i"] <= busy:
                    continue
                p = play(a, st, kind)
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
               if x["apprentissage"].get("trades", 0) >= 100 else -9)
    print("CHOISIE (sur l'apprentissage seulement)", json.dumps(best["variante"]), "-> TEST (jamais vu)",
          json.dumps(best["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/ob_retour_%s_%s.json" % (kind, tf), "w") as f:
        json.dump({"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "actifs": len(bars), "choisie": best["variante"], "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
