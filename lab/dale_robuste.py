"""Robustesse du modele Trader Dale (lab/experts.py), seul modele positif sur les deux periodes (18 nov.).
Grille : jambe de tendance 5 / 10 / 20 jours, stop 0,2 / 0,3 / 0,5 ATR jour, objectif 1,5 / 2 / 3R, duree maximale
3 jours ou 100 % intraday (ferme a 20 h UTC le jour de l'execution). Resultats par annee et par paire pour la variante
de reference (10 jours, 0,3 ATR, 2R, 3 jours). Choix sur les 60 % anciens, verdict sur les 40 % recents.
python -m lab.dale_robuste fx 1H"""
import json
import os
import sys
import time

import numpy as np

from lab.amd_vp import profile
from lab.experts import manage
from lab.liq2liq import COST
from lab.tendance_amd import daily


def trades(a, vol, kind, jours=10, stop_atr=0.3, rr=2.0, intraday=False):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr = daily(a)
    n, fond = 20, 90
    sma = np.full(len(D), np.nan)
    cs = np.cumsum(D[:, 4])
    sma[n - 1:] = (cs[n - 1:] - np.r_[0, cs[:-n]]) / n
    out, skip_until = [], -1
    for i in range(fond + 2, len(D) - 1):
        A = atr[i]
        if not (np.isfinite(A) and A > 0) or i + 1 <= skip_until:
            continue
        d = 1 if (D[i, 4] > sma[i] and D[i, 4] > D[i - fond, 4]) else (-1 if (D[i, 4] < sma[i] and D[i, 4] < D[i - fond, 4]) else 0)
        if not d:
            continue
        leg = np.arange(st[max(0, i - jours + 1)], en[i] + 1)
        Pl = profile(h[leg], l[leg], vol[leg])
        if Pl is None:
            continue
        px = c[en[i]]
        mm = (Pl["mids"] < px - 0.2 * A) if d == 1 else (Pl["mids"] > px + 0.2 * A)
        if not mm.any():
            continue
        lvl = Pl["mids"][np.flatnonzero(mm)[Pl["prof"][mm].argmax()]]
        first = leg[: len(leg) // 2]
        if not (((h[first] < lvl + 0.1 * A).sum() >= 3) if d == 1 else ((l[first] > lvl - 0.1 * A).sum() >= 3)):
            continue
        last_ok = en[i + 1] if intraday else min(len(a) - 1, st[i + 1] + 72)
        k_in = None
        for r in range(st[i + 1], last_ok + 1):
            if intraday and hr[r] >= 20:
                break
            if (l[r] <= lvl) if d == 1 else (h[r] >= lvl):
                k_in = r
                break
        if k_in is None:
            continue
        e = lvl if ((o[k_in] > lvl) if d == 1 else (o[k_in] < lvl)) else o[k_in]
        risk = stop_atr * A
        stop, tg = e - d * risk, e + d * rr * risk
        if intraday:
            res, last = manage(a, hr, k_in, en[i + 1], d, e, stop, tg, None, limite=True)
        else:
            res, last = manage(a, hr, k_in, None, d, e, stop, tg, None, max_rows=72, limite=True)
        out.append((a[k_in, 0], a[last, 0], d, float(res - COST[kind] * e / risk)))
        skip_until = int(np.searchsorted(D[:, 0], a[last, 0] // 86400))
    return out


VARIANTES = [{"jours": j, "stop_atr": s, "rr": r, "intraday": it} for j in (5, 10, 20) for s in (0.2, 0.3, 0.5)
             for r in (1.5, 2.0, 3.0) for it in (False, True)]


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    bars, vols = {}, {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, "1H")
            if kind == "fx":
                if s not in data.VOLUME:
                    continue
                tv, vv = data.VOLUME[s]
                mp = dict(zip(tv.astype(np.int64), vv))
                v = np.array([mp.get(int(t), np.nan) for t in a[:, 0]], float)
                keep = ((a[:, 0] // 86400 + 3) % 7) < 5
                a, v = a[keep], v[keep]
            else:
                if a.shape[1] <= 5:
                    continue
                v = a[:, 5].astype(float)
            if len(a) > 24 * 150 and np.nansum(v) > 0:
                bars[s], vols[s] = a[:, :5], v
        except Exception as ex:
            print("ERREUR", s, ex)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = np.quantile(t_all, 0.6)
    print("ACTIFS", kind, len(bars), "coupure", time.strftime("%d/%m/%Y", time.gmtime(cut)), flush=True)

    def stats(x):
        if not x:
            return {"trades": 0}
        x = sorted(x, key=lambda t: t[1])
        r = np.array([t[3] for t in x])
        eq = np.cumsum(r)
        yrs = max(0.25, (x[-1][1] - x[0][0]) / 86400 / 365)
        return {"trades": len(r), "par_jour": round(len(r) / yrs / 365, 2), "reussite": round(float((r > 0).mean()), 3),
                "gain_moyen_R": round(float(r.mean()), 3), "R_par_an": round(float(r.sum()) / yrs, 1),
                "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else 0,
                "pire_baisse_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}
    res, allt = [], {}
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            tt = trades(a, vols[s], kind, **v)
            allt[(json.dumps(v), s)] = tt
            for t in tt:
                (te if t[0] >= cut else tr).append(t)
        res.append({"variante": v, "apprentissage": stats(tr), "test": stats(te)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    pos_tr = sum(1 for x in res if x["apprentissage"].get("gain_moyen_R", 0) > 0)
    pos_te = sum(1 for x in res if x["test"].get("gain_moyen_R", 0) > 0)
    print("VARIANTES POSITIVES : apprentissage", pos_tr, "/", len(res), "| test", pos_te, "/", len(res), flush=True)
    for it in (False, True):
        cand = [x for x in res if x["variante"]["intraday"] == it and x["apprentissage"].get("trades", 0) >= 150]
        best = max(cand, key=lambda x: x["apprentissage"].get("t", -9))
        print("CHOISIE", "INTRADAY" if it else "3 JOURS", json.dumps(best["variante"]), "| APPRENTISSAGE",
              json.dumps(best["apprentissage"]), "| TEST", json.dumps(best["test"]), flush=True)
        key = json.dumps(best["variante"])
        tt = [t for (k, s), v in allt.items() if k == key for t in v]
        for y in sorted({time.gmtime(t[0]).tm_year for t in tt}):
            print("   ANNEE", y, json.dumps(stats([t for t in tt if time.gmtime(t[0]).tm_year == y])), flush=True)
        for s in bars:
            st_ = stats(allt[(key, s)])
            print("   PAIRE", s, st_.get("trades"), st_.get("reussite"), st_.get("gain_moyen_R"), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/dale_robuste_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res}, f, indent=1)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
