"""Version 100 % intraday de la strategie tendance + AMD (demande du trader, 16 nov. au soir) : tout est ferme le soir.
Direction :
  - « cassure »  : comme l'indicateur MT4 (cloture du jour au-dessus du plus haut / sous le plus bas des 20 jours,
                   dans le sens de la cloture d'il y a 90 jours) -> trade le lendemain seulement ;
  - « tendance » : chaque jour ou la cloture d'hier est au-dessus (dessous) de la moyenne 20 jours ET de la cloture
                   d'il y a 90 jours -> beaucoup plus de signaux.
Entree (bougies 1 h, jours UTC) : Asie 0-7 h, manipulation 7-12 h contre la direction, puis cloture de retour dans
la fourchette -> entree a l'ouverture suivante ; « amd_sinon » : sans manipulation, entree a 16 h UTC.
Sortie le jour meme : stop k ATR jour ; moitie a t1 ATR puis stop a l'entree (t1 = 0 : pas de moitie) ; objectif
final tp ATR (0 = pas d'objectif) ; sinon fermeture a 20 h UTC (21 h Paris l'hiver). Stop teste avant l'objectif.
Test honnete : 60 % anciens pour choisir, 40 % recents jamais vus, frais deduits.   python -m lab.amd_intraday fx 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import COST
from lab.tendance_amd import daily

log = logging.getLogger("lab")
FIN = 20


def trades(a, kind, dirmode="cassure", entree="amd", k=1.0, t1=0.3, tp=0.0, n=20, fond=90):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr = daily(a)
    sma = np.full(len(D), np.nan)
    cs = np.cumsum(D[:, 4])
    sma[n - 1:] = (cs[n - 1:] - np.r_[0, cs[:-n]]) / n
    out = []
    for i in range(max(n, fond, 21) + 1, len(D) - 1):
        if not (np.isfinite(atr[i]) and atr[i] > 0):
            continue
        d = 0
        if dirmode == "cassure":
            if D[i, 4] > D[i - n:i, 2].max() and D[i, 4] > D[i - fond, 4]:
                d = 1
            elif D[i, 4] < D[i - n:i, 3].min() and D[i, 4] < D[i - fond, 4]:
                d = -1
        else:
            if D[i, 4] > sma[i] and D[i, 4] > D[i - fond, 4]:
                d = 1
            elif D[i, 4] < sma[i] and D[i, 4] < D[i - fond, 4]:
                d = -1
        if not d:
            continue
        A = atr[i]
        rows = range(st[i + 1], en[i + 1] + 1)
        asie = [r for r in rows if hr[r] < 7]
        if len(asie) < 4:
            continue
        al, ah = l[asie].min(), h[asie].max()
        m, manip = None, False
        for r in rows:
            if hr[r] < 7:
                continue
            if hr[r] >= 16:
                break
            if not manip and hr[r] < 12 and ((l[r] < al) if d == 1 else (h[r] > ah)):
                manip = True
            if manip and ((c[r] > al) if d == 1 else (c[r] < ah)) and r + 1 <= en[i + 1]:
                m = r + 1
                break
        if m is None and entree == "amd_sinon":
            m = next((r for r in rows if hr[r] == 16), None)
        if m is None or hr[m] >= FIN:
            continue
        e = o[m]
        stop, half, x = e - d * k * A, False, None
        tg1, tg = e + d * t1 * A, (e + d * tp * A if tp > 0 else None)
        last = m
        for j in range(m, en[i + 1] + 1):
            if hr[j] >= FIN:
                break
            last = j
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x = stop if j == m else (min(o[j], stop) if d == 1 else max(o[j], stop))
                break
            if t1 > 0 and not half and ((h[j] >= tg1) if d == 1 else (l[j] <= tg1)):
                half = True
                stop = e
            if tg is not None and ((h[j] >= tg) if d == 1 else (l[j] <= tg)):
                x = tg
                break
        if x is None:
            x = c[last]                                      # fermeture du soir
        risk = k * A
        r = (0.5 * t1 * A / risk + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
        out.append((a[m, 0], a[last, 0], d, float(r - COST[kind] * e / risk)))
    return out


VARIANTES = [{"dirmode": dm, "entree": en, "k": k, "t1": t1, "tp": tp}
             for dm in ("cassure", "tendance") for en in ("amd", "amd_sinon") for k in (0.5, 1.0, 2.0)
             for t1 in (0.0, 0.2, 0.4) for tp in (0.0, 1.0)]


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, "1H")
            if kind == "fx":
                a = a[((a[:, 0] // 86400 + 3) % 7) < 5]
            if len(a) > 24 * 150:
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
        yrs = max(0.25, (x[-1][1] - x[0][0]) / 86400 / 365)
        w, lo = r[r > 0], r[r <= 0]
        return {"trades": len(r), "par_jour": round(len(r) / yrs / 260, 2), "reussite": round(float((r > 0).mean()), 3),
                "gain_moyen_R": round(float(r.mean()), 3),
                "gain_moyen_gagnant": round(float(w.mean()), 3) if len(w) else 0,
                "perte_moyenne": round(float(lo.mean()), 3) if len(lo) else 0, "R_par_an": round(float(r.sum()) / yrs, 1),
                "pire_baisse_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}
    res = []
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            for t in trades(a, kind, **v):
                (te if t[0] >= cut else tr).append(t)
        res.append({"variante": v, "apprentissage": stats(tr), "test": stats(te)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    ok = [x for x in res if x["apprentissage"].get("trades", 0) >= 100 and x["apprentissage"].get("reussite", 0) >= 0.8
          and x["apprentissage"].get("gain_moyen_R", 0) > 0]
    best = max(ok or res, key=lambda x: x["apprentissage"].get("gain_moyen_R", -9))
    print("CHOISIE >= 80 % et gagnante (apprentissage)" if ok else "AUCUNE >= 80 % gagnante : meilleure gain moyen",
          json.dumps(best["variante"]), "-> TEST (jamais vu)", json.dumps(best["test"]), flush=True)
    best2 = max(res, key=lambda x: x["apprentissage"].get("R_par_an", -9e9))
    print("MEILLEURE R PAR AN (apprentissage)", json.dumps(best2["variante"]), "-> TEST (jamais vu)",
          json.dumps(best2["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_intraday_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
