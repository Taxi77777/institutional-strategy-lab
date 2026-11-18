"""Methodes des specialistes : profil de volume + AMD + liquidite (demande du trader, 18 nov.).
Sources resumees (recherche web du 18 nov.) :
  - Trader Dale (« setup n°1 ») : en tendance, zone de fort volume (profil « flexible » sur la jambe de tendance) qui a
    deja servi de resistance, cassee ; on achete le premier retour sur ce niveau.
  - Fabio Valentini (champion du monde de scalping, modele de tendance) : biais si la veille cloture au-dessus de sa
    VAH (achat) / sous sa VAL (vente) ; achat sur repli vers la VWAP de la journee pendant New York ; moitie a 1R ;
    reste vers la liquidite (plus haut de la veille / extreme du jour).
  - Fabio Valentini (retour a la moyenne) : jour en equilibre (ouverture dans la zone de valeur d'hier) ; faux depart
    au-dela du plus haut / plus bas d'hier (chasse de liquidite) puis cloture de retour dans la zone -> vers le POC.
  - ICT Power of Three complet : tendance de fond, Judas sous l'ouverture de minuit New York (5 h UTC) qui balaie le
    bas de l'Asie (liquidite) dans la killzone de Londres (7-10 h UTC), deplacement qui laisse un FVG (vide entre la
    bougie 1 et la bougie 3), entree au retour dans le FVG, stop sous le Judas, objectif 2,5R ou plus haut d'hier.
  - Confluence AMD + profil + liquidite : comme ICT mais le Judas doit aussi toucher la VAL d'hier ; objectif POC
    d'hier puis plus haut d'hier.
Bougies 1 h, volume reel (crypto) / ticks (forex). Tout est ferme a 20 h UTC (Dale : 3 jours max). Moitie a 1R puis
stop a l'entree (sauf objectif fixe en R). Frais deduits. 60 % anciens pour choisir, 40 % recents.
python -m lab.experts fx 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.amd_vp import profile
from lab.liq2liq import COST
from lab.tendance_amd import daily

log = logging.getLogger("lab")


def manage(a, hr, k_in, last_row, d, e, stop, tg, half_at, fin=20, max_rows=None, limite=False):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    risk = (e - stop) * d
    half, x, last = False, None, k_in
    end = last_row if max_rows is None else min(len(a) - 1, k_in + max_rows)
    for j in range(k_in, end + 1):
        if max_rows is None and hr[j] >= fin:
            break
        last = j
        if (l[j] <= stop) if d == 1 else (h[j] >= stop):
            x = stop if j == k_in else (min(o[j], stop) if d == 1 else max(o[j], stop))
            break
        if limite and j == k_in:                              # ordre limite : rien d'optimiste sur la bougie d'execution
            continue
        if half_at is not None and not half and ((h[j] >= half_at) if d == 1 else (l[j] <= half_at)):
            half, stop = True, e
        if tg is not None and ((h[j] >= tg) if d == 1 else (l[j] <= tg)):
            x = tg
            break
    if x is None:
        x = c[last]
    rr = (0.5 * abs(half_at - e) / risk + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
    return rr, last


def trades(a, vol, kind, modele="ICT", cible="liq", biais="tendance"):
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
        rows_i = np.arange(st[i], en[i] + 1)
        P = profile(h[rows_i], l[rows_i], vol[rows_i])
        if P is None:
            continue
        pdh, pdl = D[i, 2], D[i, 3]
        trend = 1 if (D[i, 4] > sma[i] and D[i, 4] > D[i - fond, 4]) else (-1 if (D[i, 4] < sma[i] and D[i, 4] < D[i - fond, 4]) else 0)
        rows = np.arange(st[i + 1], en[i + 1] + 1)
        if len(rows) < 16:
            continue
        cost = COST[kind]
        # ------------------------------------------------------------------ Trader Dale
        if modele == "DALE":
            d = trend
            if not d:
                continue
            leg = np.arange(st[max(0, i - 9)], en[i] + 1)          # jambe de tendance : 10 jours
            Pl = profile(h[leg], l[leg], vol[leg])
            if Pl is None:
                continue
            px = c[en[i]]
            mm = (Pl["mids"] < px - 0.2 * A) if d == 1 else (Pl["mids"] > px + 0.2 * A)
            if not mm.any():
                continue
            kk = np.flatnonzero(mm)[Pl["prof"][mm].argmax()]
            lvl = Pl["mids"][kk]
            first = leg[: len(leg) // 2]                           # le niveau a d'abord bloque le prix (resistance)
            if not (((h[first] < lvl + 0.1 * A).sum() >= 3) if d == 1 else ((l[first] > lvl - 0.1 * A).sum() >= 3)):
                continue
            k_in = None
            for r in range(st[i + 1], min(len(a) - 1, st[i + 1] + 72)):   # ordre limite valable 3 jours
                if (l[r] <= lvl) if d == 1 else (h[r] >= lvl):
                    k_in = r
                    break
            if k_in is None:
                continue
            e = lvl if ((o[k_in] > lvl) if d == 1 else (o[k_in] < lvl)) else o[k_in]
            stop = e - d * 0.3 * A
            tg = (e + d * 2 * 0.3 * A) if cible == "2R" else (max(pdh, e + d * 0.3 * A) if d == 1 else min(pdl, e - d * 0.3 * A))
            rr, last = manage(a, hr, k_in, None, d, e, stop, tg, e + d * 0.3 * A if cible != "2R" else None, max_rows=72, limite=True)
            out.append((a[k_in, 0], a[last, 0], d, float(rr - cost * e / (0.3 * A))))
            skip_until = int(np.searchsorted(D[:, 0], a[last, 0] // 86400))
            continue
        # ------------------------------------------------------------------ Fabio : tendance (VWAP)
        if modele == "FABIO_T":
            d = 1 if D[i, 4] > P["vah"] else (-1 if D[i, 4] < P["val"] else 0)
            if biais == "tendance" and d != trend:
                continue
            if not d:
                continue
            pv = cv = 0.0
            k_in, low_pb = None, None
            for r in rows:
                tp = (h[r] + l[r] + c[r]) / 3
                vv = vol[r] if np.isfinite(vol[r]) else 0.0
                pv += tp * vv
                cv += vv
                if cv <= 0:
                    continue
                vw = pv / cv
                if 13 <= hr[r] < 17:
                    touch = (l[r] <= vw) if d == 1 else (h[r] >= vw)
                    hold = (c[r] > vw) if d == 1 else (c[r] < vw)
                    if touch and hold and r + 1 <= rows[-1]:
                        k_in, low_pb = r + 1, (l[r] if d == 1 else h[r])
                        break
            if k_in is None or hr[k_in] >= 20:
                continue
            e = o[k_in]
            stop = low_pb - d * 0.05 * A
            risk = (e - stop) * d
            if risk <= 0.05 * A or risk > 1.5 * A:
                continue
            tg = e + d * 2 * risk if cible == "2R" else (pdh if d == 1 else pdl)
            if (tg - e) * d < risk:
                tg = e + d * risk
            rr, last = manage(a, hr, k_in, rows[-1], d, e, stop, tg, e + d * risk if cible != "2R" else None)
            out.append((a[k_in, 0], a[last, 0], d, float(rr - cost * e / risk)))
            continue
        # ------------------------------------------------------------------ Fabio : retour a la moyenne
        if modele == "FABIO_MR":
            op = o[rows[0]]
            if not (P["val"] <= op <= P["vah"]):
                continue
            k_in = None
            for d in (1, -1):
                ext = None
                for r in rows:
                    if hr[r] < 7 or hr[r] >= 17:
                        continue
                    w_ = l[r] if d == 1 else h[r]
                    if ext is None and ((w_ < pdl) if d == 1 else (w_ > pdh)):
                        ext = w_
                    elif ext is not None:
                        ext = min(ext, w_) if d == 1 else max(ext, w_)
                    if ext is not None and ((c[r] > P["val"]) if d == 1 else (c[r] < P["vah"])) and r + 1 <= rows[-1]:
                        k_in = r + 1
                        break
                if k_in is not None:
                    break
            if k_in is None or hr[k_in] >= 20 or (biais == "tendance" and trend and d != trend):
                continue
            e = o[k_in]
            stop = ext - d * 0.05 * A
            risk = (e - stop) * d
            if risk <= 0.05 * A or risk > 1.5 * A:
                continue
            tg = P["poc"] if cible == "liq" else e + d * 2 * risk
            if (tg - e) * d < 0.5 * risk:
                continue
            rr, last = manage(a, hr, k_in, rows[-1], d, e, stop, tg, None)
            out.append((a[k_in, 0], a[last, 0], d, float(rr - cost * e / risk)))
            continue
        # ------------------------------------------------------------------ ICT Power of Three (+ confluence profil)
        d = trend
        if not d:
            continue
        asie = rows[hr[rows] < 7]
        r_mid = next((r for r in rows if hr[r] == 5), None)
        if len(asie) < 4 or r_mid is None:
            continue
        al, ah = l[asie].min(), h[asie].max()
        mo = o[r_mid]                                          # ouverture de minuit New York
        lvl = al if d == 1 else ah
        if modele == "CONF":
            lvl = min(al, P["val"]) if d == 1 else max(ah, P["vah"])
        judas, k_in, fvg = None, None, None
        rr_ = [r for r in rows if 7 <= hr[r] < 16]
        for idx, r in enumerate(rr_):
            w_ = l[r] if d == 1 else h[r]
            if judas is None:
                if hr[r] < 10 and ((w_ < lvl and w_ < mo) if d == 1 else (w_ > lvl and w_ > mo)):
                    judas = w_
                continue
            judas = min(judas, w_) if d == 1 else max(judas, w_)
            # FVG : vide entre la bougie r-2 et la bougie r (deplacement dans le sens d)
            if fvg is None and r - 2 >= rows[0]:
                if d == 1 and l[r] > h[r - 2] and c[r - 1] > o[r - 1]:
                    fvg = (h[r - 2], l[r])                     # zone a racheter : de h[r-2] a l[r]
                elif d == -1 and h[r] < l[r - 2] and c[r - 1] < o[r - 1]:
                    fvg = (l[r - 2], h[r])
                continue
            if fvg is not None and ((l[r] <= fvg[1]) if d == 1 else (h[r] >= fvg[1])):
                k_in = r
                break
        if k_in is None or hr[k_in] >= 20:
            continue
        e = fvg[1] if ((o[k_in] > fvg[1]) if d == 1 else (o[k_in] < fvg[1])) else o[k_in]
        stop = judas - d * 0.05 * A
        risk = (e - stop) * d
        if risk <= 0.05 * A or risk > 1.5 * A:
            continue
        if cible == "2R":
            tg, half_at = e + d * 2.5 * risk, None
        else:
            tg = pdh if d == 1 else pdl
            if modele == "CONF" and (P["poc"] - e) * d >= risk:
                tg = P["poc"] if (pdh - e) * d < risk else tg
            if (tg - e) * d < 1.5 * risk:
                tg = e + d * 1.5 * risk
            half_at = e + d * risk
        rr, last = manage(a, hr, k_in, rows[-1], d, e, stop, tg, half_at, limite=True)
        out.append((a[k_in, 0], a[last, 0], d, float(rr - cost * e / risk)))
    return out


VARIANTES = ([{"modele": "DALE", "cible": c} for c in ("2R", "liq")]
             + [{"modele": "FABIO_T", "cible": c, "biais": b} for c in ("2R", "liq") for b in ("profil", "tendance")]
             + [{"modele": "FABIO_MR", "cible": c, "biais": b} for c in ("liq", "2R") for b in ("aucun", "tendance")]
             + [{"modele": m, "cible": c} for m in ("ICT", "CONF") for c in ("2R", "liq")])


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
    with open("resultats/experts_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
