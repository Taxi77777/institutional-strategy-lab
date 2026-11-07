"""Modele « chiffres ronds » (Osler, Fed de New York, 2005) : les stops sont groupes juste au-dela des niveaux en 00 / 50,
les prises de benefice pile sur le 00.
  A. CASCADE : une bougie cloture au-dela d'un chiffre rond -> les stops se declenchent, on suit le mouvement
     (entree a l'ouverture suivante, stop de l'autre cote du niveau, sortie apres 1 a 2 h).
  B. REJET : le prix touche un 00 sans cloturer au-dela -> prises de benefice, on joue le retournement (30 min).
Meme methode honnete que les autres tests : 60 % anciens pour choisir la variante, 40 % recents (jamais vus) pour
le verdict.   python -m lab.ronds fx 15M   -> resultats/ronds_fx_15M.json"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import COST, _atr

log = logging.getLogger("lab")


def level_step(price):
    """Pas du « 00 » : 0,01 pour EURUSD (100 pips), 1 pour USDJPY, 10 pour l'or, 1000 pour le BTC..."""
    return 10.0 ** (np.floor(np.log10(price)) - 2)


def signals(a, atr, mode, ronds="00", stop_atr=0.5, kz=True):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((ts % 86400) // 3600).astype(int)
    out = []
    for i in range(20, len(a) - 1):
        if not np.isfinite(atr[i]) or atr[i] <= 0 or (kz and not 7 <= hr[i] < 17):
            continue
        st = level_step(c[i]) * (0.5 if ronds == "00+50" else 1.0)
        if mode == "cascade":
            lo_lv = np.floor(c[i - 1] / st) * st            # niveau juste au-dessus de la cloture precedente
            up = lo_lv + st
            if c[i - 1] <= up < c[i]:                        # cloture au-dessus d'un chiffre rond : stops d'achat
                out.append((i, 1, up - stop_atr * atr[i]))
                continue
            dn = np.ceil(c[i - 1] / st) * st - st
            if c[i] < dn <= c[i - 1]:
                out.append((i, -1, dn + stop_atr * atr[i]))
        else:                                                # rejet : seulement les 00 (prises de benefice)
            st0 = level_step(c[i])
            up = np.ceil(c[i - 1] / st0) * st0
            if h[i] >= up - 0.05 * atr[i] and c[i] < up - 0.2 * atr[i] and c[i - 1] < up:
                out.append((i, -1, up + stop_atr * atr[i]))
                continue
            dn = np.floor(c[i - 1] / st0) * st0
            if l[i] <= dn + 0.05 * atr[i] and c[i] > dn + 0.2 * atr[i] and c[i - 1] > dn:
                out.append((i, 1, dn - stop_atr * atr[i]))
    return out


def play(a, i, d, sl, h_bars, kind):
    """Entree a l'ouverture suivante ; moitie a 1 fois le risque puis stop a l'entree ; sortie apres h_bars."""
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    k0 = i + 1
    e = o[k0]
    risk = (e - sl) * d
    if risk <= 0:
        return None
    tp1, half, stop = e + d * risk, False, sl
    cost = COST[kind] * e / risk
    for k in range(k0, min(k0 + h_bars, len(a))):
        if (l[k] <= stop) if d == 1 else (h[k] >= stop):
            return (0.5 + 0.5 * (stop - e) * d / risk if half else (stop - e) * d / risk) - cost, half, k
        if not half and ((h[k] >= tp1) if d == 1 else (l[k] <= tp1)):
            half, stop = True, e
    k = min(k0 + h_bars, len(a)) - 1
    r = (0.5 + 0.5 * (c[k] - e) * d / risk) if half else (c[k] - e) * d / risk
    return r - cost, half or r > 0, k


VARIANTES = ([{"mode": "cascade", "ronds": r, "stop_atr": s, "h": hb} for r in ("00", "00+50") for s in (0.5, 1.0)
              for hb in (4, 8)] +
             [{"mode": "rejet", "ronds": "00", "stop_atr": s, "h": hb} for s in (0.3, 0.6) for hb in (2, 4)])


def run(kind, tf="15M"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, tf)
            if len(a) > 2000:
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
            for i, d, sl in signals(a, atr, v["mode"], v["ronds"], v["stop_atr"], kz=(kind == "fx")):
                if i <= busy:
                    continue
                p = play(a, i, d, sl, v["h"], kind)
                if p is None:
                    continue
                r, win, end = p
                busy = end
                (te if a[i, 0] >= cut else tr).append((r, win))

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
    with open("resultats/ronds_%s_%s.json" % (kind, tf), "w") as f:
        json.dump({"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "actifs": len(bars), "choisie": best["variante"], "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "15M")
