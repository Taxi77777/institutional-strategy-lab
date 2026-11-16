"""AMD + volume + entree precise en 15 min (demande du trader, 16 nov. : AMD intraday au resultat exceptionnel).
Sens = biais journalier : « tendance » (cloture > SMA20 et > cloture d'il y a 90 j) ou « cassure » (cloture au-dessus
du plus haut des 20 jours, dans le sens de la cloture d'il y a 90 j) ; trade le lendemain seulement.
Bougies 15 min, jours UTC : Asie 0-7 h = accumulation ; manipulation entre 7 et 12 h = le prix passe sous le bas de
l'Asie (achat ; l'inverse pour une vente) ; retour = premiere cloture 15 min de nouveau dans la fourchette (avant 15 h).
Confirmation par le volume (volume rapporte a la mediane du MEME quart d'heure sur les 20 jours precedents, donc sans
l'effet d'ouverture de Londres) :
  aucun    : pas de filtre (reference) ;
  pic      : pic de volume >= 2 x mediane sur une bougie de la meche (de la cassure jusqu'au retour) ;
  reprise  : bougie de retour avec volume >= 1,5 x mediane ;
  pic_rep  : les deux ;
  pression : bougie de retour cloturant dans le tiers haut (achat) de sa bougie avec volume >= 1,5 x mediane.
Volume : ticks Dukascopy (forex, data.VOLUME), volume Coinbase (crypto).
Entree a l'ouverture de la bougie suivante ; stop structurel sous la meche + 0,03 ATR jour ; objectif 1R, 2R, « asie »
(haut de l'Asie, au moins 1R) ou « veille » (haut d'hier, au moins 1R) ; sauf 1R : moitie a 1R puis stop a l'entree ;
tout ferme a 20 h UTC. Stop teste avant l'objectif sur chaque bougie, frais COST deduits.
Test honnete : 60 % anciens pour choisir (t >= 2, >= 150 trades), 40 % recents jamais vus pour le verdict.
python -m lab.amd_volume fx 15M"""
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
BUF = 0.03


def vol_ratio(a, jours=20):
    """Volume / mediane du meme quart d'heure sur les `jours` jours precedents (sans la bougie courante)."""
    import pandas as pd
    slot = ((a[:, 0] % 86400) // 900).astype(int)
    v = pd.Series(a[:, 5].astype(float))
    med = v.groupby(slot).transform(lambda x: x.shift(1).rolling(jours, min_periods=jours // 2).median()).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(med > 0, a[:, 5] / med, np.nan)
    return r


def trades(a, kind, biais="tendance", conf="aucun", cible="2R", vr=None, kpic=2.0, krep=1.5):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    if vr is None:
        vr = vol_ratio(a)
    D, st, en, atr = daily(a)
    n, fond = 20, 90
    sma = np.full(len(D), np.nan)
    cs = np.cumsum(D[:, 4])
    sma[n - 1:] = (cs[n - 1:] - np.r_[0, cs[:-n]]) / n
    out = []
    for i in range(fond + 2, len(D) - 1):
        A = atr[i]
        if not (np.isfinite(A) and A > 0) or D[i + 1, 0] != D[i, 0] + 1 and not (kind == "fx" and D[i + 1, 0] - D[i, 0] <= 3):
            continue
        if biais == "tendance":
            d = 1 if (D[i, 4] > sma[i] and D[i, 4] > D[i - fond, 4]) else (-1 if (D[i, 4] < sma[i] and D[i, 4] < D[i - fond, 4]) else 0)
        else:
            d = 1 if (D[i, 4] > D[i - n:i, 2].max() and D[i, 4] > D[i - fond, 4]) else \
                (-1 if (D[i, 4] < D[i - n:i, 3].min() and D[i, 4] < D[i - fond, 4]) else 0)
        if not d:
            continue
        rows = list(range(st[i + 1], en[i + 1] + 1))
        if len(rows) < 48:
            continue
        asie = [r for r in rows if hr[r] < 7]
        if len(asie) < 16:
            continue
        al, ah = l[asie].min(), h[asie].max()
        pdh, pdl = D[i, 2], D[i, 3]
        ext, k_in, pic = None, None, 0.0
        for r in rows:
            if hr[r] < 7:
                continue
            if hr[r] >= 12 and ext is None:
                break
            if hr[r] >= 15:
                break
            worst = l[r] if d == 1 else h[r]
            if ext is None and ((worst < al) if d == 1 else (worst > ah)):
                ext = worst
            elif ext is not None:
                ext = min(ext, worst) if d == 1 else max(ext, worst)
            if ext is None:
                continue
            if np.isfinite(vr[r]):
                pic = max(pic, vr[r])
            if (c[r] > al) if d == 1 else (c[r] < ah):
                rg = h[r] - l[r]
                pos = ((c[r] - l[r]) / rg if d == 1 else (h[r] - c[r]) / rg) if rg > 0 else 0.5
                vrep = vr[r] if np.isfinite(vr[r]) else 0.0
                ok = {"aucun": True, "pic": pic >= kpic, "reprise": vrep >= krep,
                      "pic_rep": pic >= kpic and vrep >= krep, "pression": pos >= 2 / 3 and vrep >= krep}[conf]
                if ok:
                    k_in = r + 1
                break                                         # un seul retour par jour : pas de seconde chance
        if k_in is None or k_in > rows[-1] or hr[k_in] >= FIN:
            continue
        e = o[k_in]
        stop = ext - d * BUF * A
        risk = (e - stop) * d
        if risk <= 0.03 * A or risk > 1.5 * A:
            continue
        if cible in ("asie", "veille"):
            tg = (ah if d == 1 else al) if cible == "asie" else (pdh if d == 1 else pdl)
            if (tg - e) * d < risk:
                tg = e + d * risk
        else:
            tg = e + d * risk * (1.0 if cible == "1R" else 2.0)
        half_at = None if cible == "1R" else e + d * risk
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
            x = c[last]                                       # fermeture du soir (20 h UTC)
        rr = (0.5 + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
        out.append((a[k_in, 0], a[last, 0], d, float(rr - COST[kind] * e / risk)))
    return out


VARIANTES = [{"biais": b, "conf": cf, "cible": t} for b in ("tendance", "cassure")
             for cf in ("aucun", "pic", "reprise", "pic_rep", "pression") for t in ("1R", "2R", "asie", "veille")]


def charger(kind, tf):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = np.array(data.long_bars(s, kind, tf), dtype=float)
            if kind == "fx":                                  # volume des ticks garde a part par data.dukascopy
                vt, vv = data.VOLUME.get(s, (None, None))
                if vt is not None and len(vt):
                    k = np.clip(np.searchsorted(vt, a[:, 0]), 0, len(vt) - 1)
                    a[:, 5] = np.where(vt[k] == a[:, 0], vv[k], np.nan)
                a = a[((a[:, 0] // 86400 + 3) % 7) < 5]
            if a.shape[1] < 6 or not np.nansum(a[:, 5]) > 0:
                log.warning("%s : pas de volume", s)
                continue
            if len(a) > 96 * 150:
                bars[s] = a[:, :6]
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    return bars


def run(kind, tf="15M", bars=None):
    bars = bars if bars is not None else charger(kind, tf)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = np.quantile(t_all, 0.6)
    print("ACTIFS", kind, len(bars), "du", time.strftime("%d/%m/%Y", time.gmtime(t_all.min())), "coupure",
          time.strftime("%d/%m/%Y", time.gmtime(cut)), "au", time.strftime("%d/%m/%Y", time.gmtime(t_all.max())), flush=True)
    vrs = {s: vol_ratio(a) for s, a in bars.items()}
    jours = 365 if kind == "cr" else 260

    def stats(x):
        if not x:
            return {"trades": 0}
        x = sorted(x, key=lambda t: t[1])
        r = np.array([t[3] for t in x])
        eq = np.cumsum(r)
        yrs = max(0.25, (x[-1][1] - x[0][0]) / 86400 / 365)
        return {"trades": len(r), "par_jour": round(len(r) / yrs / jours, 2), "reussite": round(float((r > 0).mean()), 3),
                "gain_moyen_R": round(float(r.mean()), 3), "R_par_an": round(float(r.sum()) / yrs, 1),
                "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else 0,
                "pire_baisse_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}
    res = []
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            for t in trades(a, kind, vr=vrs[s], **v):
                (te if t[0] >= cut else tr).append(t)
        res.append({"variante": v, "apprentissage": stats(tr), "test": stats(te)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    ok = [x for x in res if x["apprentissage"].get("trades", 0) >= 150 and x["apprentissage"].get("t", 0) >= 2]
    print("VARIANTES t >= 2 sur l'apprentissage :", len(ok), "/", len(res), flush=True)
    for x in sorted(ok, key=lambda x: -x["apprentissage"]["t"])[:8]:
        print("   RETENUE", json.dumps(x["variante"]), "| APPRENTISSAGE", json.dumps(x["apprentissage"]), "| TEST",
              json.dumps(x["test"]), flush=True)
    best = max(res, key=lambda x: x["apprentissage"].get("t", -99) if x["apprentissage"].get("trades", 0) >= 150 else -99)
    print("MEILLEURE t (apprentissage, >= 150 trades)", json.dumps(best["variante"]), "| APPRENTISSAGE",
          json.dumps(best["apprentissage"]), "| TEST", json.dumps(best["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_volume_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "coupure": int(cut),
                   "variantes": res}, f, indent=1)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "15M")
