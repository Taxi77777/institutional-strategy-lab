"""Modele combine « esprit bancaire » (demande du trader, 7 nov.) : liquidite + order block + logique institutionnelle.
Les banques ont besoin de contreparties : elles poussent le prix au-dela d'une zone de liquidite connue (haut/bas
asiatique, de Londres, de la veille, de la semaine) pour declencher les stops, s'y remplissent, puis lancent
l'impulsion dans l'autre sens (cassure de structure). La derniere bougie inverse avant l'impulsion est leur order
block : au premier retour dessus, elles completent leurs positions. L'objectif est la liquidite d'en face.

Regles : 1) prise d'une zone (meche au-dela, les 3 bougies d'avant ne l'avaient pas touchee) ; 2) dans les 8 bougies,
impulsion >= 1 ATR qui cloture au-dela de la structure (plus haut / bas des 7 bougies precedant la prise) ;
3) order block = derniere bougie inverse entre la prise et l'impulsion ; 4) entree au premier retour sur l'order
block (bord ou milieu) dans les 32 bougies, annulee si l'objectif est atteint avant ; 5) stop derriere la meche de
la prise (+0,1 ATR) ; objectif = zone de liquidite en face (>= 1 fois le risque) ; moitie au premier objectif puis
stop a l'entree. Option « achat en zone basse / vente en zone haute » (premium / discount de la veille).
Regle stricte : l'objectif n'est jamais compte sur la bougie du remplissage, le stop oui.
Test honnete : 60 % anciens pour choisir, 40 % recents (jamais vus) pour le verdict.   python -m lab.banque fx 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import BAS, COST, HAUTS, _atr, pools
from lab.ob_retour import play

log = logging.getLogger("lab")


def setups(a, P, atr, entry="bord", pd=False, zones="toutes", kz=True):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((ts % 86400) // 3600).astype(int)
    day = (ts // 86400).astype(int)
    n, out, taken = len(a), [], set()
    for i in range(10, n - 3):
        if not np.isfinite(atr[i]) or atr[i] <= 0 or (kz and not (7 <= hr[i] < 11 or 12 <= hr[i] < 16)):
            continue
        for d, names in ((1, BAS), (-1, HAUTS)):
            for nm in names:
                if zones == "grosses" and nm not in ("pdh", "pdl", "pwh", "pwl"):
                    continue
                lv = P[nm][i]
                if not np.isfinite(lv) or (day[i], nm) in taken:
                    continue
                if not ((l[i] < lv and l[i - 3:i].min() >= lv) if d == 1 else (h[i] > lv and h[i - 3:i].max() <= lv)):
                    continue
                taken.add((day[i], nm))
                ref = h[i - 6:i + 1].max() if d == 1 else l[i - 6:i + 1].min()
                ext = l[i] if d == 1 else h[i]
                for j in range(i + 1, min(i + 9, n - 2)):
                    ext = min(ext, l[j]) if d == 1 else max(ext, h[j])
                    if not (d * (c[j] - o[j]) >= atr[j] and ((c[j] > ref) if d == 1 else (c[j] < ref))):
                        continue
                    ob = next((m for m in range(j - 1, i - 1, -1) if d * (c[m] - o[m]) < 0), i)
                    zlo, zhi = l[ob], h[ob]
                    e = (zhi if d == 1 else zlo) if entry == "bord" else (zlo + zhi) / 2
                    sl = ext - 0.1 * atr[j] if d == 1 else ext + 0.1 * atr[j]
                    risk = (e - sl) * d
                    if risk < 0.2 * atr[j]:
                        break
                    if pd and np.isfinite(P["pdh"][j]):
                        mid = (P["pdh"][j] + P["pdl"][j]) / 2
                        if (e - mid) * d > 0:                     # achat en zone haute / vente en zone basse : non
                            break
                    tg = [P[t][j] for t in (HAUTS if d == 1 else BAS)
                          if np.isfinite(P[t][j]) and (P[t][j] - e) * d >= risk]
                    if not tg:
                        break
                    tg = min(tg, key=lambda x: (x - e) * d)
                    for m in range(j + 1, min(j + 33, n)):
                        if (l[m] <= e) if d == 1 else (h[m] >= e):
                            if not kz or 7 <= hr[m] < 17:
                                out.append({"i": m, "dir": d, "entry": float(e), "stop": float(sl), "target": float(tg)})
                            break
                        if (h[m] >= tg) if d == 1 else (l[m] <= tg):
                            break                                 # objectif atteint sans retour : rate
                    break
    return out


VARIANTES = [{"entry": e, "pd": p, "zones": z, "tp1": t} for e in ("bord", "milieu") for p in (False, True)
             for z in ("toutes", "grosses") for t in (0.5, 1.0)]


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, tf)
            if len(a) > 1500:
                bars[s] = (a, _atr(a), pools(a))
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([b[0][:, 0] for b in bars.values()])
    cut = np.quantile(t_all, 0.6)
    dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
    res = []
    for v in VARIANTES:
        tr, te = [], []
        for s, (a, atr, P) in bars.items():
            busy = -1
            for st in setups(a, P, atr, v["entry"], v["pd"], v["zones"], kz=(kind == "fx")):
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
    with open("resultats/banque_%s_%s.json" % (kind, tf), "w") as f:
        json.dump({"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "actifs": len(bars), "choisie": best["variante"], "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
