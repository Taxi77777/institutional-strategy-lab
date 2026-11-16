"""« Booster l'AMD gagnant » (demande du trader, 16 nov. : 90 % de reussite ET de gros gains).
Base : lab/tendance_amd.py (cassure 20 j dans le sens de la tendance 90 j, entree AMD le lendemain, sinon 16 h UTC,
stop 4 ATR, moitie a 0,5 ATR puis stop a l'entree, stop suiveur journalier) - reproduite a l'identique ici.
1. Gestion amelioree (24 variantes) : t1 0,5 / 1 ATR, stop 3 / 4 ATR, stop suiveur cloture journaliere ou plus bas
   des 5 / 10 derniers jours (chandelier), 2e entree (pyramide) sur une nouvelle AMD dans le sens du trade une fois
   la position sans risque (stop a l'entree). Le resultat d'un trade est la somme des deux unites en R (risque initial).
2. Meta-etiquetage : mesures connues a l'entree (taille de l'Asie / ATR, profondeur et heure de la manipulation, force
   de la cassure, pente SMA 20, ATR 20 / ATR 100, rendement BTC 7 j, jour de semaine, AMD ou 16 h, jours depuis la
   derniere cassure...) -> HistGradientBoosting ; cibles r > 0 et r > 0,5 R. Modele appris sur les 70 % anciens de
   l'apprentissage, seuil choisi sur les 30 % suivants (toujours dans l'apprentissage), verdict sur le test.
Test honnete : coupure temporelle 60 / 40 ; tout choix (variante, seuil) fait sur l'apprentissage seulement ;
frais COST deduits.   python -m lab.amd_boost cr 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import COST
from lab.tendance_amd import daily

log = logging.getLogger("lab")

NOMS = ["asie_atr", "manip_prof", "manip_heure", "amd", "heure_entree", "cassure_atr", "pente_sma20", "atr_ratio",
        "btc_7j", "btc_7j_sens", "jour_sem", "jours_derniere_cassure", "ecart_entree", "tendance90", "veille_atr",
        "sens", "pos_asie"]


def _sma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.cumsum(np.r_[0.0, x])
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def trades(a, kind, n=20, fond=90, k=4.0, t1=0.5, trail="cloture", pyr=False, btc=None):
    """Meme logique que tendance_amd.trades(entree='amd_sinon') + options de gestion + mesures a l'entree.
    Renvoie (t_entree, t_sortie, sens, r, mesures)."""
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr = daily(a)
    nd = len(D)
    is_close = np.zeros(len(a), bool)
    is_close[en] = True
    dayi = np.repeat(np.arange(nd), en - st + 1)
    # ATR 100 jours, SMA 20 des clotures, fourchette Asie de chaque jour
    pc = np.r_[D[0, 4], D[:-1, 4]]
    tr = np.maximum(D[:, 2] - D[:, 3], np.maximum(abs(D[:, 2] - pc), abs(D[:, 3] - pc)))
    atr100 = np.full(nd, np.nan)
    cs = np.cumsum(tr)
    atr100[100:] = (cs[100:] - cs[:-100]) / 100
    sma = _sma(D[:, 4], 20)
    asl, ash = np.full(nd, np.nan), np.full(nd, np.nan)
    for q in range(nd):
        rr = np.arange(st[q], en[q] + 1)
        rr = rr[hr[rr] < 7]
        if len(rr) >= 4:
            asl[q], ash[q] = l[rr].min(), h[rr].max()
    # signaux de cassure (pour « jours depuis la derniere cassure »)
    brk = np.zeros(nd, int)
    for q in range(max(n, fond, 21) + 1, nd):
        if D[q, 4] > D[q - n:q, 2].max() and (not fond or D[q, 4] > D[q - fond, 4]):
            brk[q] = 1
        elif D[q, 4] < D[q - n:q, 3].min() and (not fond or D[q, 4] < D[q - fond, 4]):
            brk[q] = -1
    out, i = [], max(n, fond, 21) + 1
    while i < nd - 2:
        d = brk[i] if (np.isfinite(atr[i]) and atr[i] > 0) else 0
        if not d:
            i += 1
            continue
        A = atr[i]
        rows = range(st[i + 1], en[i + 1] + 1)
        m, manip, mprof, mh = None, False, 0.0, -1
        if np.isfinite(asl[i + 1]):
            al, ah = asl[i + 1], ash[i + 1]
            for r in rows:
                if hr[r] < 7:
                    continue
                if not manip and hr[r] < 12 and ((l[r] < al) if d == 1 else (h[r] > ah)):
                    manip, mh = True, hr[r]
                if manip:
                    mprof = max(mprof, (al - l[r]) if d == 1 else (h[r] - ah))
                if manip and ((c[r] > al) if d == 1 else (c[r] < ah)) and r + 1 < len(a):
                    m = r + 1
                    break
                if hr[r] >= 16:
                    break
        amd = m is not None
        if m is None:
            m = next((r for r in rows if hr[r] >= 16), None)
        if m is None or m >= len(a) - 1:
            i += 1
            continue
        e = o[m]
        risk = k * A
        # mesures connues a l'entree (jour i clos, bougies du jour i+1 jusqu'a m-1)
        prev = np.flatnonzero(brk[:i] == d)
        bt = np.nan
        if btc is not None:
            b1, b0 = btc.get(int(D[i, 0])), btc.get(int(D[i, 0]) - 7)
            if b1 and b0:
                bt = b1 / b0 - 1
        lvl = D[i - n:i, 2].max() if d == 1 else D[i - n:i, 3].min()
        asz = (ash[i + 1] - asl[i + 1]) if np.isfinite(asl[i + 1]) else np.nan
        f = [asz / A, mprof / A, mh, float(amd), hr[m], d * (D[i, 4] - lvl) / A,
             d * (sma[i] - sma[i - 5]) / A if np.isfinite(sma[i - 5]) else np.nan,
             A / atr100[i] if np.isfinite(atr100[i]) else np.nan, bt, d * bt, (D[i + 1, 0] + 3) % 7,
             i - prev[-1] if len(prev) else 999, d * (e - D[i, 4]) / A, d * (D[i, 4] / D[i - fond, 4] - 1),
             d * (D[i, 4] - D[i - 1, 4]) / A, d,
             ((e - asl[i + 1]) / asz if d == 1 else (ash[i + 1] - e) / asz) if np.isfinite(asz) and asz > 0 else np.nan]
        stop, tp1, half, best = e - d * risk, e + d * t1 * A, False, e
        # 2e unite (pyramide)
        e2, stop2, x2, pend, man2, qman = None, None, None, False, False, -1
        x, j = None, m
        while j < len(a):
            q = dayi[j]
            if pend:
                e2, pend = o[j], False
                stop2 = max(e2 - risk, stop) if d == 1 else min(e2 + risk, stop)
            if e2 is not None and x2 is None and ((l[j] <= stop2) if d == 1 else (h[j] >= stop2)):
                x2 = (min(o[j], stop2) if d == 1 else max(o[j], stop2)) if j > m else stop2
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x = min(o[j], stop) if (d == 1 and j > m) else (max(o[j], stop) if j > m else stop)
                break
            if not half and ((h[j] >= tp1) if d == 1 else (l[j] <= tp1)):
                half = True
                stop = max(stop, e) if d == 1 else min(stop, e)
            # nouvelle AMD dans le sens du trade (jours suivants) -> 2e entree a l'ouverture suivante
            if pyr and half and e2 is None and q > i + 1 and np.isfinite(asl[q]) and 7 <= hr[j] < 16:
                if qman != q:
                    qman, man2 = q, False
                if not man2 and hr[j] < 12 and ((l[j] < asl[q]) if d == 1 else (h[j] > ash[q])):
                    man2 = True
                if man2 and ((c[j] > asl[q]) if d == 1 else (c[j] < ash[q])) and j + 1 < len(a):
                    pend = True
            if is_close[j]:
                best = max(best, c[j]) if d == 1 else min(best, c[j])
                if trail == "cloture":
                    cand = best - d * risk
                else:
                    nn = int(trail[2:])
                    cand = D[max(0, q - nn + 1):q + 1, 3].min() if d == 1 else D[max(0, q - nn + 1):q + 1, 2].max()
                stop = max(stop, cand) if d == 1 else min(stop, cand)
                if e2 is not None and x2 is None:
                    stop2 = max(stop2, stop) if d == 1 else min(stop2, stop)
            j += 1
        if x is None:
            break
        r = (0.5 * t1 * A / risk + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
        r -= COST[kind] * e / risk
        if e2 is not None:
            if x2 is None:
                x2 = x
            r += (x2 - e2) * d / risk - COST[kind] * e2 / risk
        out.append((a[m, 0], a[j, 0], d, float(r), f))
        i = int(np.searchsorted(D[:, 0], a[j, 0] // 86400)) + 1
    return out


VARIANTES = [{"k": k, "t1": t, "trail": tr, "pyr": p} for k in (4.0, 3.0) for t in (0.5, 1.0)
             for tr in ("cloture", "ll5", "ll10") for p in (False, True)]   # 24 ; la 1re = base tendance_amd


def stats(x):
    if not x:
        return {"trades": 0}
    x = sorted(x, key=lambda t: t[1])
    r = np.array([t[3] for t in x])
    eq = np.cumsum(r)
    yrs = max(0.25, (x[-1][1] - min(t[0] for t in x)) / 86400 / 365)
    w, lo = r[r > 0], r[r <= 0]
    return {"trades": len(r), "trades_an": round(len(r) / yrs, 1), "reussite": round(float((r > 0).mean()), 3),
            "gain_moyen_R": round(float(r.mean()), 3),
            "gain_moyen_gagnant": round(float(w.mean()), 3) if len(w) else 0,
            "perte_moyenne": round(float(lo.mean()), 3) if len(lo) else 0, "R_par_an": round(float(r.sum()) / yrs, 1),
            "t": round(float(r.mean() / (r.std() + 1e-12) * np.sqrt(len(r))), 2),
            "pire_baisse_R": round(float(np.max(np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq])), 1)}


def meta(tr, te, cible):
    """Meta-etiquetage : modele sur les 70 % anciens de l'apprentissage, seuil sur les 30 % suivants."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = sorted(tr, key=lambda t: t[0])
    if len(tr) < 100:
        return None
    X = np.array([t[4] for t in tr], float)
    cols = np.flatnonzero(np.isfinite(X).any(axis=0))      # ex. rendement BTC absent en forex
    X = X[:, cols]
    y = np.array([t[3] > cible for t in tr])
    s = int(len(tr) * 0.7)
    if y[:s].all() or not y[:s].any():
        return None
    mdl = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0).fit(X[:s], y[:s])
    pv = mdl.predict_proba(X[s:])[:, 1]
    rv = np.array([t[3] for t in tr[s:]])
    bestq, bests = None, -np.inf
    for qq in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        th = float(np.quantile(pv, qq)) if qq > 0 else -1.0
        kk = rv[pv >= th] if qq > 0 else rv
        if len(kk) < max(20, 0.2 * len(rv)):
            continue
        sc = kk.mean() * np.sqrt(len(kk)) / (kk.std() + 1e-9)
        if sc > bests:
            bests, bestq = sc, (qq, th)
    qq, th = bestq
    val = [t for t, p in zip(tr[s:], pv) if p >= th]
    pt = mdl.predict_proba(np.array([t[4] for t in te], float)[:, cols])[:, 1] if te else np.array([])
    kept = [t for t, p in zip(te, pt) if p >= th]
    return {"quantile": qq, "seuil": round(th, 4), "val": val, "test": kept}


def charger(kind, tf):
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
    return bars


def run(kind, tf="1H"):
    tf = (tf or "1H").upper()
    bars = charger(kind, tf)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = np.quantile(t_all, 0.6)
    print("ACTIFS", kind, len(bars), "du", time.strftime("%d/%m/%Y", time.gmtime(t_all.min())), "coupure",
          time.strftime("%d/%m/%Y", time.gmtime(cut)), "au", time.strftime("%d/%m/%Y", time.gmtime(t_all.max())), flush=True)
    btc = None
    if kind == "cr" and "BTC" in bars:
        Db = daily(bars["BTC"])[0]
        btc = {int(dd): float(cc) for dd, cc in zip(Db[:, 0], Db[:, 4])}
    res, splits = [], {}
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            for t in trades(a, kind, btc=btc, **v):
                (te if t[0] >= cut else tr).append(t)
        name = "gestion " + json.dumps(v)
        splits[name] = (tr, te)
        res.append({"variante": name, "apprentissage": stats(tr), "test": stats(te)})
        print("VARIANTE", name, "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    base = res[0]
    print("BASE (tendance_amd seul) | APPRENTISSAGE", json.dumps(base["apprentissage"]), "| TEST",
          json.dumps(base["test"]), flush=True)

    def critere(x):
        a_ = x["apprentissage"]
        return a_.get("R_par_an", -1e9) if a_.get("trades", 0) >= 30 and a_.get("reussite", 0) >= 0.85 else \
            -1e6 + a_.get("gain_moyen_R", -9)
    best_g = max(res, key=critere)
    # meta-etiquetage sur la base et sur la meilleure gestion (choisie sur l'apprentissage)
    cibles = [res[0]["variante"]] + ([best_g["variante"]] if best_g is not res[0] else [])
    for nm in cibles:
        tr, te = splits[nm]
        for cible in (0.0, 0.5):
            try:
                mm = meta(tr, te, cible)
            except Exception as ex:
                print("ERREUR meta", nm, repr(ex), flush=True)
                mm = None
            if mm is None:
                continue
            name = "meta r>%s q%.1f sur %s" % (cible, mm["quantile"], nm)
            res.append({"variante": name, "seuil": mm["seuil"], "apprentissage": stats(mm["val"]),
                        "apprentissage_note": "30 % finaux de l'apprentissage (hors modele)", "test": stats(mm["test"])})
            print("VARIANTE", name, "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
                  json.dumps(res[-1]["test"]), flush=True)
    print("VARIANTES testees :", len(res), flush=True)
    ret = max(res, key=critere)
    print("RETENUE (choisie sur l'apprentissage seulement)", ret["variante"], "| APPRENTISSAGE",
          json.dumps(ret["apprentissage"]), "| TEST (jamais vu)", json.dumps(ret["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_boost_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "base": base, "retenue": ret, "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "1H")
