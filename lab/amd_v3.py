"""AMD intraday v3 (demande du trader, 16 nov. soir : « je te parle de AMD », resultat exceptionnel).
Difference avec les tests precedents : stop STRUCTUREL juste derriere la meche de manipulation (pas k ATR fixe), donc
gain / risque bien plus grand ; objectifs sur la liquidite opposee. Trois modeles, sens = tendance journaliere :
  P3  Power of 3 journalier (ICT) : accumulation autour de l'ouverture du jour (0 h UTC ou minuit New York 5 h UTC),
      manipulation = faux depart contre la tendance (Judas) d'au moins m ATR sous l'ouverture (achat), puis une bougie 1 h
      cloture de nouveau au-dessus de l'ouverture -> entree ; stop sous le plus bas de la manipulation.
  NY  AMD de New York : Londres 7-12 h UTC = accumulation ; 12-15 h UTC le prix passe sous le bas de Londres (achat) ;
      retour dans la fourchette -> entree ; stop sous la meche.
  PDL Liquidite de la veille : entre 7 h et 15 h UTC le prix passe sous le plus bas d'hier (achat) puis cloture
      au-dessus -> entree ; stop sous la meche.
Objectif : « 1R », « 2R » ou « liq » (plus haut d'hier pour un achat, au moins 1R) ; moitie a 1R puis stop a l'entree
pour « liq » ; tout est ferme a 20 h UTC. Frais deduits. 60 % anciens pour choisir, 40 % recents jamais vus.
python -m lab.amd_v3 cr 1H"""
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


def trades(a, kind, modele="P3", biais="tendance", cible="2R", m=0.25, ouv=0, buf=0.05):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr = daily(a)
    n, fond = 20, 90
    sma = np.full(len(D), np.nan)
    cs = np.cumsum(D[:, 4])
    sma[n - 1:] = (cs[n - 1:] - np.r_[0, cs[:-n]]) / n
    out = []
    for i in range(fond + 2, len(D) - 1):
        A = atr[i]
        if not (np.isfinite(A) and A > 0):
            continue
        if biais == "tendance":
            d = 1 if (D[i, 4] > sma[i] and D[i, 4] > D[i - fond, 4]) else (-1 if (D[i, 4] < sma[i] and D[i, 4] < D[i - fond, 4]) else 0)
        elif biais == "cassure":
            d = 1 if (D[i, 4] > D[i - n:i, 2].max() and D[i, 4] > D[i - fond, 4]) else \
                (-1 if (D[i, 4] < D[i - n:i, 3].min() and D[i, 4] < D[i - fond, 4]) else 0)
        else:                                               # « veille » : sens de la bougie d'hier
            d = 1 if D[i, 4] > D[i, 1] else -1
        if not d:
            continue
        rows = list(range(st[i + 1], en[i + 1] + 1))
        if len(rows) < 16:
            continue
        pdh, pdl = D[i, 2], D[i, 3]
        ent, ext, k_in = None, None, None
        if modele == "P3":
            r0 = next((r for r in rows if hr[r] == ouv), None)
            if r0 is None:
                continue
            op = o[r0]
            ext_ = None
            for r in rows:
                if hr[r] < ouv:
                    continue
                if hr[r] >= 15:
                    break
                worst = l[r] if d == 1 else h[r]
                if ext_ is None and (op - worst) * d >= m * A:
                    ext_ = worst
                elif ext_ is not None:
                    ext_ = min(ext_, worst) if d == 1 else max(ext_, worst)
                if ext_ is not None and ((c[r] > op) if d == 1 else (c[r] < op)):
                    k_in, ext = r + 1, ext_
                    break
        else:
            if modele == "NY":
                lond = [r for r in rows if 7 <= hr[r] < 12]
                if len(lond) < 4:
                    continue
                lo_, hi_ = l[lond].min(), h[lond].max()
                h_a, h_b = 12, 15
            else:
                lo_, hi_ = pdl, pdh
                h_a, h_b = 7, 15
            ext_ = None
            for r in rows:
                if hr[r] < h_a:
                    continue
                if hr[r] >= h_b and ext_ is None:
                    break
                if hr[r] >= 16:
                    break
                worst = l[r] if d == 1 else h[r]
                if ext_ is None and ((worst < lo_) if d == 1 else (worst > hi_)):
                    ext_ = worst
                elif ext_ is not None:
                    ext_ = min(ext_, worst) if d == 1 else max(ext_, worst)
                if ext_ is not None and ((c[r] > lo_) if d == 1 else (c[r] < hi_)):
                    k_in, ext = r + 1, ext_
                    break
        if k_in is None or k_in > rows[-1] or hr[k_in] >= FIN:
            continue
        e = o[k_in]
        stop = ext - d * buf * A
        risk = (e - stop) * d
        if risk <= 0.05 * A or risk > 1.5 * A:              # stop absurde (trop pres ou trop loin)
            continue
        if cible == "liq":
            tg = pdh if d == 1 else pdl
            if (tg - e) * d < risk:
                tg = e + d * risk
            half_at = e + d * risk
        else:
            tg = e + d * risk * (1.0 if cible == "1R" else 2.0)
            half_at = None
        half, x, last = False, None, k_in
        for j in range(k_in, rows[-1] + 1):
            if hr[j] >= FIN:
                break
            last = j
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x = stop if j == k_in else (min(o[j], stop) if d == 1 else max(o[j], stop))
                break
            if half_at is not None and not half and ((h[j] >= half_at) if d == 1 else (l[j] <= half_at)):
                half, stop = True, e
            if (h[j] >= tg) if d == 1 else (l[j] <= tg):
                x = tg
                break
        if x is None:
            x = c[last]
        rr = (0.5 + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
        out.append((a[k_in, 0], a[last, 0], d, float(rr - COST[kind] * e / risk)))
    return out


VARIANTES = ([{"modele": "P3", "biais": b, "cible": t, "m": m, "ouv": ov} for b in ("tendance", "cassure", "veille")
              for t in ("1R", "2R", "liq") for m in (0.1, 0.25, 0.5) for ov in (0, 5)]
             + [{"modele": md, "biais": b, "cible": t} for md in ("NY", "PDL") for b in ("tendance", "cassure", "veille")
                for t in ("1R", "2R", "liq")])


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
                bars[s] = a[:, :5]
        except Exception as ex:
            log.warning("%s : %s", s, ex)
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
            for t in trades(a, kind, **v):
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
    with open("resultats/amd_v3_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
