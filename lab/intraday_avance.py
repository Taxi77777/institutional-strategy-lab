"""Intraday avance (demande du trader, 16 nov. soir : « trouve jusqu'a ce que tu trouves »). 4 familles jamais testees :
  A. Momentum intraday (Gao, Han, Li & Zhou 2018 ; Wen et al. 2022 pour le bitcoin) : le rendement du debut de journee
     (0 h -> H UTC) predit celui de la fin de journee (H -> 24 h UTC crypto, H -> 20 h UTC forex).
  B. Retard des altcoins sur le bitcoin (lead-lag) : apres une heure ou le BTC bouge de plus de z ecarts-types, on
     joue les altcoins qui n'ont pas encore suivi, dans le sens du BTC, pendant h heures.
  C. Classement des cryptos (transversal) : toutes les h heures, achat du dixieme le plus fort (ou le plus faible) sur
     les L heures passees et vente de l'autre dixieme ; le sens est choisi sur l'apprentissage.
  D. Cassure de la fourchette d'ouverture (ORB) : fourchette des n premieres heures (0 h UTC crypto, 7 h UTC Londres
     forex), cassure dans le sens de la tendance journaliere ou sans filtre, stop de l'autre cote, sortie 20 h UTC.
Test honnete : choix sur les 60 % anciens, verdict sur les 40 % recents ; frais 0,12 % crypto / 0,015 % forex.
python -m lab.intraday_avance cr 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

log = logging.getLogger("lab")
COSTS = {"cr": 0.0012, "fx": 0.00015}


def st(r, ts, yrs):
    if len(r) == 0:
        return {"trades": 0}
    r = np.asarray(r, float)
    o = np.argsort(ts)
    eq = np.cumsum(r[o])
    return {"trades": int(len(r)), "par_jour": round(len(r) / yrs / 365, 2), "reussite": round(float((r > 0).mean()), 3),
            "gain_moyen_pb": round(float(r.mean() * 1e4), 2),
            "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else 0,
            "total_pct": round(float(r.sum() * 100), 1),
            "pire_baisse_pct": round(float((np.maximum.accumulate(eq) - eq).max() * 100), 1)}


def panel(bars):
    """Matrice des prix horaires (lignes = heures communes, colonnes = actifs) : ouverture et cloture."""
    T = np.unique(np.concatenate([a[:, 0] for a in bars.values()]))
    ix = {t: k for k, t in enumerate(T)}
    O = np.full((len(T), len(bars)), np.nan)
    C = np.full_like(O, np.nan)
    H = np.full_like(O, np.nan)
    L = np.full_like(O, np.nan)
    for j, a in enumerate(bars.values()):
        rows = np.array([ix[t] for t in a[:, 0]])
        O[rows, j], H[rows, j], L[rows, j], C[rows, j] = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    return T, O, H, L, C


def run(kind, tf="1H"):
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
    syms = list(bars)
    cost = COSTS[kind]
    T, O, H, L, C = panel(bars)
    cut = float(np.quantile(T, 0.6))
    y_tr, y_te = (cut - T[0]) / 86400 / 365, (T[-1] - cut) / 86400 / 365
    hr = ((T % 86400) // 3600).astype(int)
    day = (T // 86400).astype(np.int64)
    print("ACTIFS", kind, len(syms), "du", time.strftime("%d/%m/%Y", time.gmtime(T[0])), "coupure",
          time.strftime("%d/%m/%Y", time.gmtime(cut)), flush=True)
    res = {}

    def verdict(name, cand):
        """cand : liste de (nom_variante, r, ts). Choix sur l'apprentissage (gain moyen, t >= 2), verdict sur le test."""
        rows = []
        for nm, r, ts in cand:
            r, ts = np.asarray(r, float), np.asarray(ts, float)
            m = ts < cut
            rows.append({"variante": nm, "apprentissage": st(r[m], ts[m], y_tr), "test": st(r[~m], ts[~m], y_te)})
        ok = [x for x in rows if x["apprentissage"].get("trades", 0) >= 200 and x["apprentissage"].get("t", 0) >= 2]
        best = max(ok, key=lambda x: x["apprentissage"]["t"]) if ok else None
        res[name] = {"variantes": rows, "choisie": best}
        print("FAMILLE", name, "|", len(ok), "/", len(rows), "variantes t >= 2 sur l'apprentissage", flush=True)
        if best:
            print("   CHOISIE", best["variante"], "| APPRENTISSAGE", json.dumps(best["apprentissage"]), "| TEST",
                  json.dumps(best["test"]), flush=True)
        for x in sorted(ok, key=lambda x: -x["apprentissage"]["t"])[:5]:
            print("   top", x["variante"], "| TEST", json.dumps(x["test"]), flush=True)

    # ---- A. momentum intraday
    end = 24 if kind == "cr" else 20
    cand = []
    starts = {}
    for k in range(len(T)):
        starts.setdefault(day[k], {})[hr[k]] = k
    for Hh in ((4, 8, 12, 16, 20) if kind == "cr" else (7, 12, 16)):
        for sens in (1, -1):
            r, ts = [], []
            for dd, hh in starts.items():
                if 0 not in hh or Hh not in hh or (end - 1) not in hh:
                    continue
                k0, kH, kE = hh[0], hh[Hh], hh[end - 1]
                f = C[kH - 1] / O[k0] - 1
                g = C[kE] / O[kH] - 1
                ok = np.isfinite(f) & np.isfinite(g) & (f != 0)
                for j in np.flatnonzero(ok):
                    r.append(sens * np.sign(f[j]) * g[j] - cost)
                    ts.append(T[kH])
            cand.append(("debut 0-%dh -> fin %d-%dh %s" % (Hh, Hh, end, "meme sens" if sens > 0 else "a l'envers"), r, ts))
    verdict("A_momentum_intraday", cand)

    # ---- B. lead-lag BTC -> altcoins (crypto)
    if kind == "cr" and "BTC" in syms:
        b = syms.index("BTC")
        R1 = C / O - 1
        sd = np.full(len(T), np.nan)
        rb = R1[:, b]
        for k in range(200, len(T)):
            w = rb[k - 168:k]
            sd[k] = np.nanstd(w)
        cand = []
        for z in (2.0, 3.0):
            for h in (1, 2, 4):
                r, ts = [], []
                for k in np.flatnonzero(np.abs(rb) > z * sd)[:-h - 1]:
                    d = np.sign(rb[k])
                    for j in range(len(syms)):
                        if j == b or not np.isfinite(R1[k, j]) or d * R1[k, j] > 0.5 * abs(rb[k]):
                            continue
                        e, x = O[k + 1, j], C[k + h, j]
                        if np.isfinite(e) and np.isfinite(x) and e > 0:
                            r.append(d * (x / e - 1) - cost)
                            ts.append(T[k + 1])
                cand.append(("BTC > %.0f ecarts-types, altcoins en retard, %d h" % (z, h), r, ts))
        verdict("B_retard_altcoins", cand)

    # ---- C. classement transversal
    if len(syms) >= 10:
        cand = []
        q = max(1, len(syms) // 10)
        for Lb in (4, 24, 72):
            for h in (4, 8, 24):
                for sens in (1, -1):
                    r, ts = [], []
                    for k in range(Lb, len(T) - h, h):
                        past = C[k - 1] / O[k - Lb] - 1
                        fut = C[k + h - 1] / O[k] - 1
                        ok = np.flatnonzero(np.isfinite(past) & np.isfinite(fut))
                        if len(ok) < 2 * q + 2:
                            continue
                        o = ok[np.argsort(past[ok])]
                        top, bot = o[-q:], o[:q]
                        r.append(sens * (fut[top].mean() - fut[bot].mean()) / 2 - cost)
                        ts.append(T[k])
                    cand.append(("classement %d h, garde %d h, %s" % (Lb, h, "achat des plus forts" if sens > 0 else "achat des plus faibles"), r, ts))
        verdict("C_classement", cand)

    # ---- D. ORB
    cand = []
    h0 = 0 if kind == "cr" else 7
    dO = {}
    for k in range(len(T)):
        dO.setdefault(day[k], []).append(k)
    days = sorted(dO)
    dclose = {}
    for dd in days:
        ks = dO[dd]
        dclose[dd] = C[ks[-1]]
    for n in (1, 2, 4):
        for filt in ("aucun", "tendance"):
            r, ts = [], []
            for di, dd in enumerate(days):
                if di < 21:
                    continue
                ks = [k for k in dO[dd] if h0 <= hr[k] < 20]
                if len(ks) < n + 3 or hr[ks[0]] != h0:
                    continue
                rng = ks[:n]
                hi, lo = np.nanmax(H[rng], axis=0), np.nanmin(L[rng], axis=0)
                prev = np.array([dclose[days[di - q2]] for q2 in (1, 20)])
                for j in range(len(syms)):
                    if not (np.isfinite(hi[j]) and np.isfinite(lo[j]) and hi[j] > lo[j]):
                        continue
                    tr_ = 0
                    if filt == "tendance":
                        p1, p20 = prev[0, j], prev[1, j]
                        if not (np.isfinite(p1) and np.isfinite(p20)):
                            continue
                        tr_ = 1 if p1 > p20 else -1
                    pos, e, stop, x, t_in = 0, 0, 0, None, 0
                    for k in ks[n:]:
                        if pos == 0:
                            if H[k, j] > hi[j] and tr_ >= 0:
                                pos, e, stop, t_in = 1, max(O[k, j], hi[j]), lo[j], T[k]
                            elif L[k, j] < lo[j] and tr_ <= 0:
                                pos, e, stop, t_in = -1, min(O[k, j], lo[j]), hi[j], T[k]
                            if pos == 0:
                                continue
                        if (L[k, j] <= stop) if pos == 1 else (H[k, j] >= stop):
                            x = stop
                            break
                    if pos == 0:
                        continue
                    if x is None:
                        x = C[ks[-1], j]
                    if np.isfinite(x):
                        r.append(pos * (x / e - 1) - cost)
                        ts.append(t_in)
            cand.append(("fourchette %d h des %d h UTC, filtre %s" % (n, h0, filt), r, ts))
    verdict("D_cassure_ouverture", cand)

    os.makedirs("resultats", exist_ok=True)
    with open("resultats/intraday_avance_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "familles": res},
                  f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
