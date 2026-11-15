"""Strategie de la video envoyee par le trader (15 nov., « seulement trois bougies », ecran triple d'Alexander Elder) :
  1. Jour (direction) : la bougie d'hier est passee sous le plus bas de la veille (PDL) puis a cloture au-dessus :
     vendeurs pieges -> on ne cherche que des achats. Inverse au-dessus du plus haut de la veille (PDH) -> ventes.
  2. 1H (setup) : pendant la journee, le prix passe sous le plus bas des N dernieres bougies 1H (prise de liquidite).
  3. 15 min (entree) : apres cette prise, une bougie cloture au-dessus du plus haut de la derniere bougie rouge
     (cassure) ; cette bougie rouge est l'order block ; entree au retour sur l'order block ; stop proche, sous la
     meche de la prise (+0,1 ATR) ; TP1 un peu sous le plus haut 1H (moitie, puis stop a l'entree), TP2 sur le plus
     haut 1H. Inverse pour les ventes.
Variante « sans filtre jour » pour mesurer ce que le filtre apporte. Regle stricte (objectif jamais compte sur la
bougie d'entree). Test honnete : 60 % anciens pour choisir, 40 % recents (jamais vus).   python -m lab.elder3 fx"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import COST, _atr

log = logging.getLogger("lab")


def play2(a, s, kind, max_bars=64):
    """Moitie a TP1 (puis stop a l'entree), moitie a TP2 ; objectif jamais compte sur la bougie d'entree."""
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    m, d, e, sl, t1, t2 = s["i"], s["dir"], s["entry"], s["stop"], s["tp1"], s["target"]
    if (o[m] - e) * d < 0:
        e = o[m]
    risk = (e - sl) * d
    if risk <= 0 or (t1 - e) * d <= 0:
        return None
    r1 = (t1 - e) * d / risk
    half, stop = False, sl
    cost = COST[kind] * e / risk
    for k in range(m, min(m + max_bars, len(a))):
        if (l[k] <= stop) if d == 1 else (h[k] >= stop):
            return ((0.5 * r1 if half else (stop - e) * d / risk) + (0.5 * (stop - e) * d / risk if half else 0)) - cost, half, k
        if k == m:
            continue
        if not half and ((h[k] >= t1) if d == 1 else (l[k] <= t1)):
            half, stop = True, e
        if (h[k] >= t2) if d == 1 else (l[k] <= t2):
            return 0.5 * r1 + 0.5 * (t2 - e) * d / risk - cost, True, k
    k = min(m + max_bars, len(a)) - 1
    r = (0.5 * r1 + 0.5 * (c[k] - e) * d / risk) if half else (c[k] - e) * d / risk
    return r - cost, half, k


def setups(a, atr, filtre=True, n1h=4, entry="bord", rr_min=1.0, kz=True):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    day = (ts // 86400).astype(int)
    hour = (ts // 3600).astype(int)
    hr = ((ts % 86400) // 3600).astype(int)
    D = {}
    for i, d in enumerate(day):
        D.setdefault(d, []).append(i)
    dk = sorted(D)
    dinfo = {d: (h[D[d]].max(), l[D[d]].min(), c[D[d][-1]]) for d in dk}
    H = {}
    for i, x in enumerate(hour):
        H.setdefault(x, []).append(i)
    hinfo = {x: (h[H[x]].max(), l[H[x]].min()) for x in H}
    out = []
    for k in range(2, len(dk)):
        dd, y, z = dk[k], dk[k - 1], dk[k - 2]
        yh, yl, yc = dinfo[y]
        zh, zl, _ = dinfo[z]
        biais = []
        if not filtre:
            biais = [1, -1]
        else:
            if yl < zl and yc > zl:
                biais.append(1)                                # vendeurs pieges sous le PDL
            if yh > zh and yc < zh:
                biais.append(-1)                               # acheteurs pieges au-dessus du PDH
            if len(biais) == 2:
                biais = []
        rows = D[dd]
        for d in biais:
            done = False
            for idx, i in enumerate(rows[:-2]):
                if done:
                    break
                if kz and not 7 <= hr[i] < 17:
                    continue
                A = atr[i]
                if not np.isfinite(A) or A <= 0:
                    continue
                prevh = [hour[i] - j for j in range(1, n1h + 1) if hour[i] - j in hinfo]
                if len(prevh) < n1h:
                    continue
                L1 = min(hinfo[x][1] for x in prevh)
                H1 = max(hinfo[x][0] for x in prevh)
                if not ((l[i] < L1) if d == 1 else (h[i] > H1)):
                    continue
                # prise de liquidite 1H en i : cherche la cassure 15 min
                ext = l[i] if d == 1 else h[i]
                n = len(a)
                for j in range(i + 1, min(i + 9, n - 2)):
                    ext = min(ext, l[j]) if d == 1 else max(ext, h[j])
                    ob = next((m for m in range(j - 1, i - 1, -1) if d * (c[m] - o[m]) < 0), None)
                    if ob is None:
                        continue
                    if not ((c[j] > h[ob]) if d == 1 else (c[j] < l[ob])):
                        continue
                    zlo, zhi = l[ob], h[ob]
                    e = (zhi if d == 1 else zlo) if entry == "bord" else (zlo + zhi) / 2
                    sl = ext - 0.1 * A if d == 1 else ext + 0.1 * A
                    t2 = H1 if d == 1 else L1
                    risk = (e - sl) * d
                    if risk < 0.2 * A or (t2 - e) * d < rr_min * risk:
                        done = True
                        break
                    t1 = e + 0.8 * (t2 - e)                    # « un peu en dessous » du plus haut 1H
                    for m in range(j + 1, min(j + 17, n)):
                        if (l[m] <= e) if d == 1 else (h[m] >= e):
                            if not kz or 7 <= hr[m] < 17:
                                out.append({"i": m, "dir": d, "entry": float(e), "stop": float(sl),
                                            "tp1": float(t1), "target": float(t2)})
                            break
                        if (h[m] >= t2) if d == 1 else (l[m] <= t2):
                            break
                        if (c[m] < sl) if d == 1 else (c[m] > sl):
                            break
                    done = True
                    break
    out.sort(key=lambda x: x["i"])
    return out


VARIANTES = [{"filtre": f, "n1h": n, "entry": e, "rr_min": r} for f in (True, False) for n in (4, 8)
             for e in ("bord", "milieu") for r in (1.0, 2.0)]


def run(kind, tf="15M"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, "15M")
            if kind == "fx":
                a = a[((a[:, 0] // 86400 + 3) % 7) < 5]
            if len(a) > 1500:
                bars[s] = (a, _atr(a))
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    print("ACTIFS", kind, len(bars), flush=True)
    t_all = np.concatenate([b[0][:, 0] for b in bars.values()])
    cut = np.quantile(t_all, 0.6)
    dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
    res = []
    for v in VARIANTES:
        tr, te = [], []
        for s, (a, atr) in bars.items():
            busy = -1
            for st in setups(a, atr, v["filtre"], v["n1h"], v["entry"], v["rr_min"], kz=(kind == "fx")):
                if st["i"] <= busy:
                    continue
                p = play2(a, st, kind)
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
    with open("resultats/elder3_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "actifs": len(bars), "choisie": best["variante"], "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1])
