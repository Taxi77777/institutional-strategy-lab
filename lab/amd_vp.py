"""AMD + profil de volume, version avancee (demande du trader, 17 nov. : « AMD et volume profil seulement »).
Le volume (ticks Dukascopy en forex, volume reel en crypto) de chaque bougie 1 h est reparti sur sa fourchette de prix
pour construire des profils : POC (prix le plus traite), zone de valeur (70 % du volume, VAL / VAH).
Sens : « tendance » (cloture > moyenne 20 j et > cloture 90 j) ou « forme » (profil d'hier en P : POC dans le tiers
haut -> achat ; en b : POC dans le tiers bas -> vente).
Accumulation = Asie 0-7 h UTC (son propre profil : POC, VAL, VAH). Manipulation 7-12 h UTC contre le sens. Modeles :
  ASIE  : la manipulation sort de la zone de valeur de l'Asie, puis une cloture 1 h revient DANS la zone (acceptation)
          -> entree ; objectif VAH de l'Asie (achat) puis plus loin.
  VAL   : la manipulation doit toucher la VAL d'hier (achat) / VAH d'hier (vente) = support de valeur, puis cloture
          au-dessus du bas de l'Asie -> entree ; objectif POC d'hier ou VAH.
  HVN   : profil composite 5 jours ; la manipulation touche un noeud de fort volume (HVN) sous le prix, rejet ->
          entree ; objectif POC composite.
  NPOC  : comme ASIE, objectif = POC « nu » (jamais retouche) des 10 derniers jours dans le sens du trade.
Option volume : la meche de manipulation doit avoir un volume FAIBLE (rejet sans acceptation) ou FORT (absorption).
Stop derriere la meche (+0,05 ATR), moitie a 1R puis stop a l'entree, objectif du modele (au moins 1R), sortie 20 h UTC.
Frais deduits. 60 % anciens pour choisir (t >= 2, >= 150 trades), 40 % recents jamais vus.   python -m lab.amd_vp fx 1H"""
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
BINS = 40


def profile(h, l, v, lo=None, hi=None):
    lo = float(np.min(l)) if lo is None else lo
    hi = float(np.max(h)) if hi is None else hi
    if not (hi > lo) or np.nansum(v) <= 0:
        return None
    edges = np.linspace(lo, hi, BINS + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    prof = np.zeros(BINS)
    for hh, ll, vv in zip(h, l, v):
        if not np.isfinite(vv) or vv <= 0:
            continue
        a, b = np.searchsorted(edges, ll, "right") - 1, np.searchsorted(edges, hh, "left")
        a, b = max(0, a), min(BINS, max(b, a + 1))
        prof[a:b] += vv / (b - a)
    k = int(prof.argmax())
    tot, acc, i, j = prof.sum(), prof[k], k, k
    while acc < 0.7 * tot and (i > 0 or j < BINS - 1):
        up = prof[j + 1] if j < BINS - 1 else -1
        dn = prof[i - 1] if i > 0 else -1
        if up >= dn:
            j += 1
            acc += up
        else:
            i -= 1
            acc += dn
    return {"poc": mids[k], "val": edges[i], "vah": edges[j + 1], "prof": prof, "mids": mids}


def trades(a, vol, kind, modele="ASIE", biais="tendance", filtre_vol="aucun", buf=0.05):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr = daily(a)
    n, fond = 20, 90
    sma = np.full(len(D), np.nan)
    cs = np.cumsum(D[:, 4])
    sma[n - 1:] = (cs[n - 1:] - np.r_[0, cs[:-n]]) / n
    vmed = np.full(len(a), np.nan)                           # volume habituel de la meme heure (20 jours)
    if filtre_vol != "aucun":
        for r in range(24 * 21, len(a)):
            same = vol[r - 24 * 20:r][hr[r - 24 * 20:r] == hr[r]]
            same = same[np.isfinite(same)]
            vmed[r] = np.median(same) if len(same) else np.nan
    pd_prof = {}
    out, naked = [], []                                      # POC nus : (prix, jour)
    for i in range(fond + 2, len(D) - 1):
        A = atr[i]
        rows_i = np.arange(st[i], en[i] + 1)
        P = profile(h[rows_i], l[rows_i], vol[rows_i])
        if P is not None:
            # POC nus : retires s'ils sont touches aujourd'hui
            naked = [(p, dd) for p, dd in naked if not (l[rows_i].min() <= p <= h[rows_i].max()) and i - dd <= 10]
            naked.append((P["poc"], i))
        if P is None or not (np.isfinite(A) and A > 0):
            continue
        if biais == "tendance":
            d = 1 if (D[i, 4] > sma[i] and D[i, 4] > D[i - fond, 4]) else (-1 if (D[i, 4] < sma[i] and D[i, 4] < D[i - fond, 4]) else 0)
        else:
            pos = (P["poc"] - D[i, 3]) / max(D[i, 2] - D[i, 3], 1e-12)
            d = 1 if pos > 0.66 else (-1 if pos < 0.34 else 0)
        if not d:
            continue
        rows = np.arange(st[i + 1], en[i + 1] + 1)
        asie = rows[hr[rows] < 7]
        if len(asie) < 4:
            continue
        al, ah = l[asie].min(), h[asie].max()
        PA = profile(h[asie], l[asie], vol[asie])
        if PA is None:
            continue
        comp = None
        if modele == "HVN" and i >= 5:
            rr5 = np.arange(st[i - 4], en[i] + 1)
            comp = profile(h[rr5], l[rr5], vol[rr5])
            if comp is None:
                continue
        # niveau a balayer et niveau de reprise
        if modele in ("ASIE", "NPOC"):
            sweep_lvl = PA["val"] if d == 1 else PA["vah"]
            reclaim = sweep_lvl
        elif modele == "VAL":
            sweep_lvl = P["val"] if d == 1 else P["vah"]
            reclaim = al if d == 1 else ah
        else:
            # HVN sous l'ouverture de l'Asie (achat) : plus gros noeud parmi les prix sous le bas de l'Asie
            m_ = comp["mids"] < al if d == 1 else comp["mids"] > ah
            if not m_.any():
                continue
            kk = np.flatnonzero(m_)[comp["prof"][m_].argmax()]
            sweep_lvl = comp["mids"][kk]
            reclaim = al if d == 1 else ah
        ext, k_in = None, None
        for r in rows:
            if hr[r] < 7:
                continue
            if hr[r] >= 15:
                break
            worst = l[r] if d == 1 else h[r]
            if ext is None and hr[r] < 12 and ((worst < sweep_lvl) if d == 1 else (worst > sweep_lvl)):
                if filtre_vol != "aucun" and np.isfinite(vmed[r]) and vmed[r] > 0 and np.isfinite(vol[r]):
                    rel = vol[r] / vmed[r]
                    if (filtre_vol == "faible" and rel > 1.0) or (filtre_vol == "fort" and rel < 1.5):
                        continue
                ext = worst
            elif ext is not None:
                ext = min(ext, worst) if d == 1 else max(ext, worst)
            if ext is not None and ((c[r] > reclaim) if d == 1 else (c[r] < reclaim)):
                k_in = r + 1
                break
        if k_in is None or k_in > rows[-1] or hr[k_in] >= FIN:
            continue
        e = o[k_in]
        stop = ext - d * buf * A
        risk = (e - stop) * d
        if risk <= 0.05 * A or risk > 1.5 * A:
            continue
        if modele == "ASIE":
            tg = PA["vah"] if d == 1 else PA["val"]
        elif modele == "VAL":
            tg = P["poc"] if (P["poc"] - e) * d >= risk else (P["vah"] if d == 1 else P["val"])
        elif modele == "HVN":
            tg = comp["poc"]
        else:
            cands = [p for p, _ in naked if (p - e) * d >= risk]
            tg = min(cands, key=lambda p: abs(p - e)) if cands else None
            if tg is None:
                continue
        if (tg - e) * d < risk:
            tg = e + d * risk
        half_at = e + d * risk
        half, x, last = False, None, k_in
        for j in range(k_in, rows[-1] + 1):
            if hr[j] >= FIN:
                break
            last = j
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x = stop if j == k_in else (min(o[j], stop) if d == 1 else max(o[j], stop))
                break
            if not half and ((h[j] >= half_at) if d == 1 else (l[j] <= half_at)):
                half, stop = True, e
            if (h[j] >= tg) if d == 1 else (l[j] <= tg):
                x = tg
                break
        if x is None:
            x = c[last]
        rr = (0.5 + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
        out.append((a[k_in, 0], a[last, 0], d, float(rr - COST[kind] * e / risk)))
    return out


VARIANTES = [{"modele": m, "biais": b, "filtre_vol": f} for m in ("ASIE", "VAL", "HVN", "NPOC")
             for b in ("tendance", "forme") for f in ("aucun", "faible", "fort")]


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    bars, vols = {}, {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, "1H")
            if kind == "fx":
                if s in data.VOLUME:
                    tv, vv = data.VOLUME[s]
                    m = dict(zip(tv.astype(np.int64), vv))
                    v = np.array([m.get(int(t), np.nan) for t in a[:, 0]], float)
                else:
                    continue
                keep = ((a[:, 0] // 86400 + 3) % 7) < 5
                a, v = a[keep], v[keep]
            else:
                v = a[:, 5].astype(float) if a.shape[1] > 5 else None
                if v is None:
                    continue
            if len(a) > 24 * 200 and np.nansum(v) > 0:
                bars[s], vols[s] = a[:, :5], v
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = np.quantile(t_all, 0.6)
    print("ACTIFS", kind, len(bars), "du", time.strftime("%d/%m/%Y", time.gmtime(t_all.min())), "coupure",
          time.strftime("%d/%m/%Y", time.gmtime(cut)), flush=True)

    def stats(x):
        if not x:
            return {"trades": 0}
        x = sorted(x, key=lambda t: t[1])
        r = np.array([t[3] for t in x])
        eq = np.cumsum(r)
        yrs = max(0.25, (x[-1][1] - x[0][0]) / 86400 / 365)
        w, lo = r[r > 0], r[r <= 0]
        return {"trades": len(r), "par_jour": round(len(r) / yrs / 365, 2), "reussite": round(float((r > 0).mean()), 3),
                "gain_moyen_R": round(float(r.mean()), 3), "gain_moyen_gagnant": round(float(w.mean()), 3) if len(w) else 0,
                "perte_moyenne": round(float(lo.mean()), 3) if len(lo) else 0, "R_par_an": round(float(r.sum()) / yrs, 1),
                "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else 0,
                "pire_baisse_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}
    res = []
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            for t in trades(a, vols[s], kind, **v):
                (te if t[0] >= cut else tr).append(t)
        res.append({"variante": v, "apprentissage": stats(tr), "test": stats(te)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    ok = [x for x in res if x["apprentissage"].get("trades", 0) >= 150 and x["apprentissage"].get("t", 0) >= 2]
    print("VARIANTES t >= 2 sur l'apprentissage :", len(ok), "/", len(res), flush=True)
    for x in sorted(ok, key=lambda x: -x["apprentissage"]["t"])[:8]:
        print("   RETENUE", json.dumps(x["variante"]), "| APPRENTISSAGE", json.dumps(x["apprentissage"]), "| TEST",
              json.dumps(x["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_vp_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
