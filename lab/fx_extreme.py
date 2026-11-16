"""Forex : anomalies documentees par la recherche, jamais testees dans le labo (demande du trader, 16 nov.).
  A. Saisonnalite horaire (Breedon & Ranaldo 2013 : une devise baisse pendant les heures de bureau de son pays) :
     pour chaque paire et chaque session UTC (Asie 0-7, Londres 7-12, chevauchement 12-16, New York 16-21),
     on achete ou vend la paire a l'ouverture de la session et on sort a sa fin.
  B. Momentum de fond multi-mois (fonds de tendance / CTA) : signe du rendement sur 3, 6 ou 12 mois, position
     ajustee a la volatilite, portefeuille de toutes les paires, reajuste chaque vendredi.
  C. Fin de mois : du 2e dernier jour ouvre au 3e jour du mois suivant (flux de reequilibrage des gerants).
  D. Jour de la semaine.
  E. Retour a la moyenne apres un tres grand jour (> 2 ecarts-types) : on joue le jour suivant a l'envers.
Test honnete : la direction et le choix (t > 2 sur les 60 % anciens) sont faits sans voir les 40 % recents.
Frais : 1,5 point de base par aller-retour (2,5 pour l'or et l'argent).   python -m lab.fx_extreme fx 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

log = logging.getLogger("lab")
SESSIONS = {"asie_0_7": (0, 7), "londres_7_12": (7, 12), "chevauchement_12_16": (12, 16), "new_york_16_21": (16, 21)}


def cost(sym):
    return 2.5e-4 if sym.startswith(("XAU", "XAG")) else 1.5e-4


def daily(a):
    day = (a[:, 0] // 86400).astype(np.int64)
    cuts = np.flatnonzero(np.diff(day)) + 1
    st, en = np.r_[0, cuts], np.r_[cuts, len(a)] - 1
    keep = (en - st) >= 12                                   # jours complets seulement
    return day[st][keep], a[st, 1][keep], a[en, 4][keep]


def tstat(x):
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 10 and x.std() > 0 else 0.0


def st(x, years=None):
    if len(x) == 0:
        return {"trades": 0}
    x = np.asarray(x, float)
    eq = np.cumsum(x)
    out = {"trades": int(len(x)), "reussite": round(float((x > 0).mean()), 3),
           "gain_moyen_pb": round(float(x.mean() * 1e4), 2), "t": round(tstat(x), 2),
           "pire_baisse_pct": round(float((np.maximum.accumulate(eq) - eq).max() * 100), 2)}
    if years:
        out["rendement_pct_an"] = round(float(x.sum() * 100 / years), 2)
    return out


def select(cands):
    """cands : liste de (nom, train_returns, test_returns). Garde ceux dont t >= 2 sur l'apprentissage."""
    keep = [(n, tr, te) for n, tr, te in cands if len(tr) > 50 and tstat(tr) >= 2.0]
    tr = np.concatenate([c[1] for c in keep]) if keep else np.array([])
    te = np.concatenate([c[2] for c in keep]) if keep else np.array([])
    return keep, tr, te


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, "1H")
            a = a[((a[:, 0] // 86400 + 3) % 7) < 5]
            if len(a) > 24 * 400:
                bars[s] = a
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = float(np.quantile(t_all, 0.6))
    y_tr, y_te = (cut - t_all.min()) / 86400 / 365, (t_all.max() - cut) / 86400 / 365
    print("ACTIFS", len(bars), "du", time.strftime("%d/%m/%Y", time.gmtime(t_all.min())), "coupure",
          time.strftime("%d/%m/%Y", time.gmtime(cut)), "au", time.strftime("%d/%m/%Y", time.gmtime(t_all.max())), flush=True)
    res = {}

    def report(name, cands, extra=None):
        keep, tr, te = select(cands)
        sens = [{"choix": n, "t_apprentissage": round(tstat(a), 2), "test": st(b)} for n, a, b in keep]
        sens.sort(key=lambda z: -z["t_apprentissage"])
        res[name] = {"retenus": len(keep), "sur": len(cands), "apprentissage": st(tr), "test": st(te),
                     "detail": sens[:25]}
        if extra:
            res[name].update(extra)
        print("MODELE", name, "| retenus", len(keep), "/", len(cands), "| APPRENTISSAGE", json.dumps(st(tr)),
              "| TEST", json.dumps(st(te)), flush=True)

    # ---- A. sessions
    cands = []
    for s, a in bars.items():
        hr = ((a[:, 0] % 86400) // 3600).astype(int)
        day = a[:, 0] // 86400
        for nm, (h0, h1) in SESSIONS.items():
            i0 = np.flatnonzero(hr == h0)
            r, t0 = [], []
            for i in i0:
                j = i
                while j + 1 < len(a) and day[j + 1] == day[i] and hr[j + 1] < h1:
                    j += 1
                if hr[j] != h1 - 1:
                    continue
                r.append(a[j, 4] / a[i, 1] - 1)
                t0.append(a[i, 0])
            r, t0 = np.array(r), np.array(t0)
            if len(r) < 200:
                continue
            m = t0 < cut
            d = 1.0 if r[m].mean() > 0 else -1.0
            c = cost(s)
            cands.append(("%s %s %s" % (s, nm, "achat" if d > 0 else "vente"), d * r[m] - c, d * r[~m] - c))
    report("A_sessions", cands)

    # ---- jours
    D = {s: daily(a) for s, a in bars.items()}

    # ---- B. momentum de fond en portefeuille
    days_all = np.unique(np.concatenate([d[0] for d in D.values()]))
    idx = {d: k for k, d in enumerate(days_all)}
    C = np.full((len(days_all), len(D)), np.nan)
    for k, (s, (dy, o, c)) in enumerate(D.items()):
        C[[idx[x] for x in dy], k] = c
    for k in range(C.shape[1]):                               # report du dernier prix connu
        col = C[:, k]
        for t in range(1, len(col)):
            if np.isnan(col[t]):
                col[t] = col[t - 1]
    R = np.vstack([np.full(C.shape[1], np.nan), C[1:] / C[:-1] - 1])
    syms = list(D)
    costs = np.array([cost(s) for s in syms])
    for L in (63, 126, 252):
        pos = np.zeros_like(C)
        vol = np.full_like(C, np.nan)
        for t in range(60, len(C)):
            vol[t] = np.nanstd(R[t - 59:t + 1], axis=0) * np.sqrt(252)
        cur = np.zeros(C.shape[1])
        for t in range(L + 60, len(C)):
            if (days_all[t] + 3) % 7 == 4 or not cur.any():      # vendredi
                sig = np.sign(C[t] / C[t - L] - 1)
                w = np.where(np.isfinite(vol[t]) & (vol[t] > 0), 0.10 / np.maximum(vol[t], 1e-6), 0) / C.shape[1]
                cur = np.nan_to_num(sig * w)
            pos[t] = cur
        pr = np.nansum(pos[:-1] * np.nan_to_num(R[1:]), axis=1) - np.sum(np.abs(np.diff(pos, axis=0)) * costs, axis=1)
        t_pr = days_all[1:] * 86400.0
        m = (t_pr < cut) & (np.abs(pos[:-1]).sum(axis=1) > 0)
        mt = (t_pr >= cut) & (np.abs(pos[:-1]).sum(axis=1) > 0)

        def pstats(x, yrs):
            if len(x) < 20:
                return {}
            mo = np.add.reduceat(x, np.arange(0, len(x), 21))
            eq = np.cumsum(x)
            return {"rendement_pct_an": round(float(x.sum() * 100 / yrs), 2),
                    "sharpe": round(float(x.mean() / x.std() * np.sqrt(252)), 2) if x.std() > 0 else 0,
                    "mois_gagnants": round(float((mo > 0).mean()), 3),
                    "pire_baisse_pct": round(float((np.maximum.accumulate(eq) - eq).max() * 100), 2)}
        res["B_momentum_%dj" % L] = {"apprentissage": pstats(pr[m], y_tr), "test": pstats(pr[mt], y_te),
                                      "note": "portefeuille vise 10 % de volatilite par an"}
        print("MODELE B_momentum_%dj" % L, "| APPRENTISSAGE", json.dumps(res["B_momentum_%dj" % L]["apprentissage"]),
              "| TEST", json.dumps(res["B_momentum_%dj" % L]["test"]), flush=True)

    # ---- C. fin de mois, D. jour de semaine, E. retour a la moyenne
    cC, cD, cE = [], [], []
    for s, (dy, o, c) in D.items():
        r = c / o - 1                                         # rendement du jour (ouverture -> cloture)
        cc = cost(s)
        tm = dy * 86400.0 < cut
        mon = np.array([time.gmtime(x * 86400).tm_mon for x in dy])
        last = np.r_[mon[1:] != mon[:-1], False]               # dernier jour ouvre du mois
        pos = np.full(len(dy), 99)
        for k in np.flatnonzero(last):
            for off in range(-1, 4):
                if 0 <= k + off < len(dy):
                    pos[k + off] = off
        win = (pos <= 3)
        # trade : du 2e dernier jour au 3e jour suivant = somme des jours de la fenetre, un trade par mois
        grp, cur, rs, ts = [], None, [], []
        for k in range(len(dy)):
            if win[k]:
                if cur is None:
                    cur, t0 = 0.0, dy[k]
                cur += r[k]
            elif cur is not None:
                rs.append(cur); ts.append(t0); cur = None
        rs, ts = np.array(rs), np.array(ts) * 86400.0
        if len(rs) > 30:
            d = 1.0 if rs[ts < cut].mean() > 0 else -1.0
            cC.append(("%s fin de mois %s" % (s, "achat" if d > 0 else "vente"), d * rs[ts < cut] - cc, d * rs[ts >= cut] - cc))
        wd = (dy + 3) % 7
        for k, nm in enumerate(("lundi", "mardi", "mercredi", "jeudi", "vendredi")):
            mm = wd == k
            if mm.sum() < 100:
                continue
            d = 1.0 if r[mm & tm].mean() > 0 else -1.0
            cD.append(("%s %s %s" % (s, nm, "achat" if d > 0 else "vente"), d * r[mm & tm] - cc, d * r[mm & ~tm] - cc))
        rc = np.r_[np.nan, c[1:] / c[:-1] - 1]
        sd = np.array([np.nanstd(rc[max(1, k - 60):k]) if k > 20 else np.nan for k in range(len(rc))])
        big = np.abs(rc) > 2 * sd
        nx = np.flatnonzero(big[:-1]) + 1
        rr = -np.sign(rc[nx - 1]) * r[nx] - cc
        tt = dy[nx] * 86400.0 < cut
        cE.append(("%s apres grand jour" % s, rr[tt], rr[~tt]))
    report("C_fin_de_mois", cC)
    report("D_jour_semaine", cD)
    # E : pas de choix de direction, on garde toutes les paires (montre le resultat brut aussi)
    allE_tr = np.concatenate([x[1] for x in cE]); allE_te = np.concatenate([x[2] for x in cE])
    report("E_retour_moyenne", cE, {"toutes_paires": {"apprentissage": st(allE_tr), "test": st(allE_te)}})
    print("E toutes paires | APPRENTISSAGE", json.dumps(st(allE_tr)), "| TEST", json.dumps(st(allE_te)), flush=True)

    os.makedirs("resultats", exist_ok=True)
    with open("resultats/fx_extreme.json", "w") as f:
        json.dump({"genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "annees_apprentissage": round(y_tr, 1),
                   "annees_test": round(y_te, 1), "modeles": res}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    if sys.argv[1] != "fx":
        print("forex seulement")
        sys.exit(0)
    run("fx")
