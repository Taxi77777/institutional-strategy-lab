"""Tendance journaliere + entree AMD (demande du trader, 15 nov. : 90 % minimum, AMD integre).
Direction : cassure du plus haut (bas) des 20 derniers jours a la cloture du jour, avec ou sans tendance de fond
(cloture au-dessus / dessous de celle d'il y a 90 jours) : le seul modele gagnant de tous nos tests.
Entree le lendemain (bougies 1 h, jours UTC) :
  - « ouverture » : a l'ouverture du jour (reference) ;
  - « amd » : on attend la manipulation de Londres CONTRE la tendance (7 h - 12 h UTC, le prix passe sous le bas de
    l'Asie 0 h - 7 h pour un achat), puis une cloture 1 h de retour dans la fourchette -> entree a l'ouverture
    suivante ; sans manipulation avant 16 h UTC, pas de trade ;
  - « amd_sinon » : pareil, mais sans manipulation on entre a 16 h UTC.
Gestion : stop a k ATR jour ; moitie encaissee a t1 ATR jour puis stop a l'entree ; stop suiveur (meilleure cloture
journaliere - k ATR) mis a jour a chaque cloture du jour ; le stop est teste avant l'objectif sur chaque bougie 1 h.
Une position a la fois par actif. Test honnete : 60 % anciens pour choisir, 40 % recents.   python -m lab.tendance_amd cr"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import COST

log = logging.getLogger("lab")


def daily(a):
    ts = a[:, 0]
    day = (ts // 86400).astype(np.int64)
    cuts = np.flatnonzero(np.diff(day)) + 1
    st, en = np.r_[0, cuts], np.r_[cuts, len(a)] - 1
    D = np.c_[day[st], a[st, 1], [a[s:e + 1, 2].max() for s, e in zip(st, en)],
              [a[s:e + 1, 3].min() for s, e in zip(st, en)], a[en, 4]]
    pc = np.r_[D[0, 4], D[:-1, 4]]
    tr = np.maximum(D[:, 2] - D[:, 3], np.maximum(abs(D[:, 2] - pc), abs(D[:, 3] - pc)))
    atr = np.full(len(D), np.nan)
    cs = np.cumsum(tr)
    atr[20:] = (cs[20:] - cs[:-20]) / 20
    return D, st, en, atr


def trades(a, kind, n=20, fond=90, k=3.0, t1=0.5, entree="ouverture"):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr = daily(a)
    is_close = np.zeros(len(a), bool)
    is_close[en] = True
    out, i = [], max(n, fond, 21) + 1
    while i < len(D) - 2:
        d = 0
        if np.isfinite(atr[i]) and atr[i] > 0:
            if D[i, 4] > D[i - n:i, 2].max() and (not fond or D[i, 4] > D[i - fond, 4]):
                d = 1
            elif D[i, 4] < D[i - n:i, 3].min() and (not fond or D[i, 4] < D[i - fond, 4]):
                d = -1
        if not d:
            i += 1
            continue
        A = atr[i]
        rows = range(st[i + 1], en[i + 1] + 1)
        m = None
        if entree == "ouverture":
            m = st[i + 1]
        else:
            asie = [r for r in rows if hr[r] < 7]
            if len(asie) >= 4:
                al, ah = l[asie].min(), h[asie].max()
                manip = False
                for r in rows:
                    if hr[r] < 7:
                        continue
                    if not manip and hr[r] < 12 and ((l[r] < al) if d == 1 else (h[r] > ah)):
                        manip = True
                    if manip and ((c[r] > al) if d == 1 else (c[r] < ah)) and r + 1 < len(a):
                        m = r + 1
                        break
                    if hr[r] >= 16:
                        break
            if m is None and entree == "amd_sinon":
                m = next((r for r in rows if hr[r] >= 16), None)
        if m is None or m >= len(a) - 1:
            i += 1
            continue
        e = o[m]
        risk = k * A
        stop, tp1, half, best = e - d * risk, e + d * t1 * A, False, e
        x, j = None, m
        while j < len(a):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x = min(o[j], stop) if (d == 1 and j > m) else (max(o[j], stop) if j > m else stop)
                break
            if not half and ((h[j] >= tp1) if d == 1 else (l[j] <= tp1)):
                half = True
                stop = max(stop, e) if d == 1 else min(stop, e)
            if is_close[j]:
                best = max(best, c[j]) if d == 1 else min(best, c[j])
                stop = max(stop, best - risk) if d == 1 else min(stop, best + risk)
            j += 1
        if x is None:
            break
        r = (0.5 * t1 * A / risk + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
        out.append((a[m, 0], a[j, 0], d, float(r - COST[kind] * e / risk)))
        i = int(np.searchsorted(D[:, 0], a[j, 0] // 86400)) + 1
    return out


VARIANTES = [{"fond": f, "k": k, "t1": t, "entree": e} for f in (0, 90) for k in (3.0, 4.0, 5.0)
             for t in (0.3, 0.5, 0.75) for e in ("ouverture", "amd", "amd_sinon")]


def run(kind, tf="1H"):
    tf = (tf or "1H").upper()
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, tf)
            if kind == "fx":
                a = a[((a[:, 0] // 86400 + 3) % 7) < 5]
            if len(a) > (86400 // data.TF[tf][2]) * 150:
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
        return {"trades": len(r), "reussite": round(float((r > 0).mean()), 3), "gain_moyen_R": round(float(r.mean()), 3),
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
    ok = [x for x in res if x["apprentissage"].get("trades", 0) >= 50 and x["apprentissage"].get("reussite", 0) >= 0.9]
    best = max(ok or res, key=lambda x: x["apprentissage"].get("gain_moyen_R", -9))
    print("CHOISIE >= 90 % (sur l'apprentissage seulement)" if ok else "AUCUNE >= 90 % : meilleure", json.dumps(best["variante"]),
          "-> TEST (jamais vu)", json.dumps(best["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/tendance_amd_%s%s.json" % (kind, "" if tf == "1H" else "_" + tf), "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
