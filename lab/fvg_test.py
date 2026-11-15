"""Modeles FVG (fair value gap, ICT), demande du trader (15 nov.) : trois facons classiques de les jouer.
FVG haussier : le bas de la bougie i depasse le haut de la bougie i-2 (zone [h(i-2), l(i)]), bougie i-1 = impulsion.
  - « simple » : premier retour du prix dans le gap -> entree au bord ou au milieu (consequent encroachment) ;
  - « breakaway » (BAG) : seulement les gaps crees par une impulsion qui casse le plus haut / bas des 20 bougies ;
  - « inverse » (IFVG) : un gap traverse en cloture change de role ; au premier retest on joue le nouveau sens.
Stop : sous la bougie d'origine (i-2) pour simple / breakaway, sous le gap pour l'inverse (+0,1 ATR). Objectif : le
plus haut (bas) atteint depuis le gap (>= 1 fois le risque), sinon 2 fois le risque. Moitie au premier objectif puis
stop a l'entree ; sortie forcee apres 64 bougies. Le setup est annule si le prix cloture au-dela du gap ou atteint
l'objectif avant le retour. Regle stricte (objectif jamais compte sur la bougie d'entree). Test honnete : 60 %
anciens pour choisir, 40 % recents (jamais vus) pour le verdict.   python -m lab.fvg_test fx 15M"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import _atr
from lab.ob_retour import play

log = logging.getLogger("lab")
WAIT = 48


def setups(a, atr, typ="simple", entry="bord", gap_min=0.2, kz=True):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((ts % 86400) // 3600).astype(int)
    n, out = len(a), []
    for i in range(22, n - 2):
        A = atr[i]
        if not np.isfinite(A) or A <= 0:
            continue
        for d in (1, -1):
            if d == 1:
                lo, hi = h[i - 2], l[i]
            else:
                lo, hi = h[i], l[i - 2]
            if hi - lo < gap_min * A:
                continue
            if typ == "breakaway":
                if not ((c[i - 1] > h[i - 22:i - 1].max()) if d == 1 else (c[i - 1] < l[i - 22:i - 1].min())):
                    continue
            if typ == "inverse":
                # le gap de sens -d est traverse en cloture dans le sens d : il devient support / resistance
                inv = None
                for m in range(i + 1, min(i + 1 + WAIT, n - 1)):
                    if (c[m] > hi) if d == -1 else (c[m] < lo):
                        inv = m
                        break
                    if (c[m] < lo) if d == -1 else (c[m] > hi):
                        break
                if inv is None:
                    continue
                dd = -d                                        # nouveau sens
                e = hi if dd == 1 else lo
                sl = (lo - 0.1 * A) if dd == 1 else (hi + 0.1 * A)
                start, ext = inv + 1, (h[inv] if dd == 1 else l[inv])
            else:
                dd = d
                e = (hi if d == 1 else lo) if entry == "bord" else (lo + hi) / 2
                sl = (l[i - 2] - 0.1 * A) if d == 1 else (h[i - 2] + 0.1 * A)
                start, ext = i + 1, (h[i] if d == 1 else l[i])
            risk = (e - sl) * dd
            if risk < 0.2 * A:
                continue
            for m in range(start, min(start + WAIT, n)):
                if (l[m] <= e) if dd == 1 else (h[m] >= e):
                    if kz and not 7 <= hr[m] < 17:
                        break
                    tg = ext if (ext - e) * dd >= risk else e + dd * 2 * risk
                    out.append({"i": m, "dir": dd, "entry": float(e), "stop": float(sl), "target": float(tg)})
                    break
                if (c[m] < sl) if dd == 1 else (c[m] > sl):
                    break
                ext = max(ext, h[m]) if dd == 1 else min(ext, l[m])
    out.sort(key=lambda x: x["i"])
    return out


VARIANTES = ([{"type": t, "entry": e, "gap_min": g, "tp1": p} for t in ("simple", "breakaway") for e in ("bord", "milieu")
              for g in (0.2, 0.5) for p in (0.5, 1.0)] +
             [{"type": "inverse", "entry": "bord", "gap_min": g, "tp1": p} for g in (0.2, 0.5) for p in (0.5, 1.0)])


def run(kind, tf="15M"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, tf)
            if len(a) > 1500:
                bars[s] = (a, _atr(a))
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    print("ACTIFS", kind, tf, len(bars), flush=True)
    t_all = np.concatenate([b[0][:, 0] for b in bars.values()])
    cut = np.quantile(t_all, 0.6)
    dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
    cache, res = {}, []
    for v in VARIANTES:
        tr, te = [], []
        for s, (a, atr) in bars.items():
            k = (s, v["type"], v["entry"], v["gap_min"])
            if k not in cache:
                cache[k] = setups(a, atr, v["type"], v["entry"], v["gap_min"], kz=(kind == "fx"))
            busy = -1
            for st in cache[k]:
                if st["i"] <= busy:
                    continue
                p = play(a, st, kind, tp1_r=v["tp1"])
                if p is None:
                    continue
                r, win, end = p
                busy = end
                (te if a[st["i"], 0] >= cut else tr).append((r, win, st["dir"]))

        def stats(x, days):
            if not x:
                return {"trades": 0}
            r = np.array([y[0] for y in x])
            return {"trades": len(x), "reussite": round(float(np.mean([y[1] for y in x])), 3),
                    "gain_moyen_R": round(float(r.mean()), 3), "total_R": round(float(r.sum()), 1),
                    "par_jour": round(len(x) / max(1, days), 2),
                    "achats_R": round(float(sum(y[0] for y in x if y[2] == 1)), 1),
                    "ventes_R": round(float(sum(y[0] for y in x if y[2] == -1)), 1)}
        res.append({"variante": v, "apprentissage": stats(tr, dtr), "test": stats(te, dte)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    best = max(res, key=lambda x: x["apprentissage"].get("gain_moyen_R", -9)
               if x["apprentissage"].get("trades", 0) >= 50 else -9)
    print("CHOISIE (sur l'apprentissage seulement)", json.dumps(best["variante"]), "-> TEST (jamais vu)",
          json.dumps(best["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/fvg_test_%s_%s.json" % (kind, tf), "w") as f:
        json.dump({"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "actifs": len(bars), "choisie": best["variante"], "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "15M")
