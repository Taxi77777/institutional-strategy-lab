"""AMD intraday + tri par apprentissage automatique (meta-etiquetage, Lopez de Prado) : demande du trader, 16 nov. soir.
1. Candidats : tous les signaux AMD intraday (direction « tendance » ou « cassure », entree AMD ou 16 h UTC,
   fermeture a 20 h UTC), comme lab/amd_intraday.py.
2. Pour chaque candidat, 16 mesures connues AU MOMENT de l'entree (taille de l'Asie, profondeur et heure de la
   manipulation, force de la tendance, volatilite, jour, mouvement d'hier...).
3. Un modele (gradient boosting) apprend sur les 60 % anciens quels candidats gagnent ; le seuil de probabilite est
   choisi sur la fin de l'apprentissage (jamais sur le test) ; verdict sur les 40 % recents jamais vus.
python -m lab.amd_intraday_ml cr 1H"""
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
NOMS = ["dir", "asie_atr", "manip_prof_atr", "heure_entree", "amd", "tend90_atr", "ecart_sma20_atr", "cassure",
        "entree_vs_asie_atr", "hier_range_atr", "vol_rel", "jour", "ret5_atr", "hier_sens", "bougie_atr", "dist_h20_atr"]


def candidats(a, kind, k=1.0, t1=0.2, tp=0.0, n=20, fond=90):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr = daily(a)
    sma = np.full(len(D), np.nan)
    cs = np.cumsum(D[:, 4])
    sma[n - 1:] = (cs[n - 1:] - np.r_[0, cs[:-n]]) / n
    atr_m = np.full(len(D), np.nan)
    for i in range(120, len(D)):
        atr_m[i] = np.nanmean(atr[i - 100:i])
    X, Y = [], []
    for i in range(max(n, fond, 121) + 1, len(D) - 1):
        A = atr[i]
        if not (np.isfinite(A) and A > 0):
            continue
        brk = 1 if D[i, 4] > D[i - n:i, 2].max() else (-1 if D[i, 4] < D[i - n:i, 3].min() else 0)
        d = 1 if (D[i, 4] > sma[i] and D[i, 4] > D[i - fond, 4]) else (-1 if (D[i, 4] < sma[i] and D[i, 4] < D[i - fond, 4]) else 0)
        if not d:
            continue
        rows = range(st[i + 1], en[i + 1] + 1)
        asie = [r for r in rows if hr[r] < 7]
        if len(asie) < 4:
            continue
        al, ah = l[asie].min(), h[asie].max()
        m, manip, ext, amd = None, False, 0.0, 0
        for r in rows:
            if hr[r] < 7:
                continue
            if hr[r] >= 16:
                break
            if hr[r] < 12 and ((l[r] < al) if d == 1 else (h[r] > ah)):
                manip = True
            if manip:
                ext = max(ext, (al - l[r]) if d == 1 else (h[r] - ah))
            if manip and ((c[r] > al) if d == 1 else (c[r] < ah)) and r + 1 <= en[i + 1]:
                m, amd = r + 1, 1
                break
        if m is None:
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
                half, stop = True, e
            if tg is not None and ((h[j] >= tg) if d == 1 else (l[j] <= tg)):
                x = tg
                break
        if x is None:
            x = c[last]
        risk = k * A
        r = (0.5 * t1 * A / risk + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
        r -= COST[kind] * e / risk
        pb = m - 1
        X.append([d, (ah - al) / A, ext / A, hr[m], amd, d * (D[i, 4] - D[i - fond, 4]) / A, d * (D[i, 4] - sma[i]) / A,
                  brk * d, d * (e - (al + ah) / 2) / A, (D[i, 2] - D[i, 3]) / A,
                  A / atr_m[i] if np.isfinite(atr_m[i]) and atr_m[i] > 0 else 1.0, (D[i + 1, 0] + 3) % 7,
                  d * (D[i, 4] - D[i - 5, 4]) / A, d * np.sign(D[i, 4] - D[i, 1]), (h[pb] - l[pb]) / A,
                  d * (D[i - n:i, 2].max() - D[i, 4]) / A if d == 1 else d * (D[i - n:i, 3].min() - D[i, 4]) / A])
        Y.append((a[m, 0], a[last, 0], r, brk != 0))
    return X, Y


def stats(rows, yrs):
    if len(rows) == 0:
        return {"trades": 0}
    r = np.array([x[2] for x in sorted(rows, key=lambda t: t[1])])
    eq = np.cumsum(r)
    w, lo = r[r > 0], r[r <= 0]
    return {"trades": int(len(r)), "par_jour": round(len(r) / yrs / 260, 2), "reussite": round(float((r > 0).mean()), 3),
            "gain_moyen_R": round(float(r.mean()), 3), "gain_moyen_gagnant": round(float(w.mean()), 3) if len(w) else 0,
            "perte_moyenne": round(float(lo.mean()), 3) if len(lo) else 0, "R_par_an": round(float(r.sum()) / yrs, 1),
            "pire_baisse_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}


def run(kind, tf="1H"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, "1H")
            if kind == "fx":
                a = a[((a[:, 0] // 86400 + 3) % 7) < 5]
            if len(a) > 24 * 200:
                bars[s] = a
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut, t0, t1_ = np.quantile(t_all, 0.6), t_all.min(), t_all.max()
    y_te = (t1_ - cut) / 86400 / 365
    print("ACTIFS", kind, len(bars), "coupure", time.strftime("%d/%m/%Y", time.gmtime(cut)), flush=True)
    out = []
    for k, t1, tp in ((1.0, 0.2, 0.0), (2.0, 0.2, 0.0), (1.0, 0.0, 1.0), (0.5, 0.2, 0.0)):
        X, Y = [], []
        for s, a in bars.items():
            x, y = candidats(a, kind, k, t1, tp)
            X += x
            Y += y
        X, ts = np.array(X, float), np.array([y[0] for y in Y])
        R = np.array([y[2] for y in Y])
        o = np.argsort(ts)
        X, ts, R, Y = X[o], ts[o], R[o], [Y[j] for j in o]
        tr, te = ts < cut, ts >= cut
        # seuil choisi a l'interieur de l'apprentissage : modele sur les 70 % premiers, seuil sur les 30 % suivants
        itr = np.flatnonzero(tr)
        c2 = itr[int(len(itr) * 0.7)]
        m1 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=100,
                                            random_state=0).fit(X[:c2], R[:c2] > 0)
        pv = m1.predict_proba(X[c2:itr[-1] + 1])[:, 1]
        Rv = R[c2:itr[-1] + 1]
        best_q, best = None, -9
        for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            th = np.quantile(pv, q)
            sel = pv >= th
            if sel.sum() >= 60 and Rv[sel].mean() > best:
                best, best_q = Rv[sel].mean(), q
        m2 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=100,
                                            random_state=0).fit(X[tr], R[tr] > 0)
        ptr = m2.predict_proba(X[tr])[:, 1]
        th = np.quantile(ptr, best_q) if best_q is not None else 1.1
        pte = m2.predict_proba(X[te])[:, 1]
        sel = pte >= th
        ytest = [Y[j] for j in np.flatnonzero(te)]
        res = {"gestion": {"stop_atr": k, "moitie_atr": t1, "objectif_atr": tp},
               "tous_les_signaux_test": stats(ytest, y_te),
               "seulement_cassure_test": stats([y for y in ytest if y[3]], y_te),
               "quantile_choisi": best_q, "gain_validation_R": round(float(best), 3),
               "tries_par_le_modele_test": stats([ytest[j] for j in np.flatnonzero(sel)], y_te)}
        # seuils fixes pour voir la courbe (information, pas choisis)
        res["courbe_test"] = {str(q): stats([ytest[j] for j in np.flatnonzero(pte >= np.quantile(ptr, q))], y_te)
                              for q in (0.5, 0.8, 0.9, 0.95)}
        out.append(res)
        print("GESTION", json.dumps(res["gestion"]), "| TOUS (test)", json.dumps(res["tous_les_signaux_test"]),
              "| TRI MODELE (test, seuil", best_q, ")", json.dumps(res["tries_par_le_modele_test"]), flush=True)
        for q, v in res["courbe_test"].items():
            print("   quantile", q, json.dumps(v), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_intraday_ml_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "mesures": NOMS,
                   "resultats": out}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
