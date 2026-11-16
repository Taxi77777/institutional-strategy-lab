"""Forex intraday : reaction aux annonces economiques americaines (demande du trader, 16 nov. : intraday forex).
Les chiffres US (emploi, inflation, ventes au detail...) tombent a 8 h 30 heure de New York (13 h 30 UTC l'hiver,
12 h 30 l'ete). On mesure le mouvement de la premiere bougie 15 min (8 h 30 - 8 h 45) ; s'il depasse k fois le
mouvement habituel d'une bougie 15 min, on entre a 8 h 45 :
  « suite »  : dans le sens du premier mouvement (derive post-annonce, Andersen et al. 2003) ;
  « retour » : a l'envers (sur-reaction).
Sortie apres h heures, stop au-dela de l'extremite de la bougie de reaction (suite) ou a 1 fois la reaction (retour).
Evenements : « nfp » = premier vendredi du mois (emploi) ; « tous » = tous les jours ouvres a 8 h 30 (les jours sans
chiffre, le mouvement est petit et filtre par k). Paires contre USD + or. Frais 1,5 pb (3 pb pour l'or).
60 % anciens pour choisir, 40 % recents.   python -m lab.fx_annonces fx 15M"""
import calendar
import json
import logging
import os
import sys
import time

import numpy as np

log = logging.getLogger("lab")
PAIRES = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD")


def us_dst(t):
    y = time.gmtime(t).tm_year
    def nth_sunday(m, n):
        d = calendar.timegm((y, m, 1, 0, 0, 0, 0, 0, 0))
        wd = time.gmtime(d).tm_wday                        # lundi = 0
        return d + ((6 - wd) % 7) * 86400 + (n - 1) * 7 * 86400
    return nth_sunday(3, 2) + 7 * 3600 <= t < nth_sunday(11, 1) + 6 * 3600


def trades(a, sym, events="tous", k=2.0, h=2, sens="suite"):
    o, hi, lo, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    t = a[:, 0]
    cost = 3e-4 if sym.startswith("XAU") else 1.5e-4
    rng = np.abs(c / o - 1)
    idx = {int(x): i for i, x in enumerate(t)}
    out = []
    days = np.unique((t // 86400).astype(np.int64))
    for dd in days:
        g = time.gmtime(dd * 86400)
        if g.tm_wday > 4:
            continue
        if events == "nfp" and not (g.tm_wday == 4 and g.tm_mday <= 7):
            continue
        hh = 12 if us_dst(dd * 86400 + 12 * 3600) else 13
        k0 = idx.get(int(dd * 86400 + hh * 3600 + 30 * 60))
        if k0 is None or k0 < 400 or k0 + 1 + h * 4 >= len(t):
            continue
        typ = np.median(rng[k0 - 400:k0])
        r0 = c[k0] / o[k0] - 1
        if typ <= 0 or abs(r0) < k * typ:
            continue
        d = np.sign(r0) * (1 if sens == "suite" else -1)
        e = o[k0 + 1]
        if sens == "suite":
            stop = lo[k0] if d > 0 else hi[k0]
        else:
            stop = e - d * abs(c[k0] - o[k0])
        risk = (e - stop) * d
        if risk <= 0:
            continue
        x = None
        end = k0 + 1 + h * 4
        for j in range(k0 + 1, end):
            if (lo[j] <= stop) if d > 0 else (hi[j] >= stop):
                x = stop if j == k0 + 1 else (min(o[j], stop) if d > 0 else max(o[j], stop))
                break
        if x is None:
            x = c[end - 1]
        out.append((t[k0 + 1], (x - e) * d / e - cost, (x - e) * d / risk - cost * e / risk))
    return out


VARIANTES = [{"events": ev, "k": k, "h": h, "sens": s} for ev in ("nfp", "tous") for k in (1.5, 2.5, 4.0)
             for h in (1, 2, 4) for s in ("suite", "retour")]


def run(kind, tf="15M"):
    from lab import data
    if kind != "fx":
        print("forex seulement")
        return
    bars = {}
    for s in PAIRES:
        try:
            a = data.long_bars(s, "fx", "15M")
            if len(a) > 96 * 200:
                bars[s] = a[:, :5]
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = np.quantile(t_all, 0.6)
    print("ACTIFS", len(bars), "du", time.strftime("%d/%m/%Y", time.gmtime(t_all.min())), "coupure",
          time.strftime("%d/%m/%Y", time.gmtime(cut)), flush=True)

    def st(x):
        if not x:
            return {"trades": 0}
        p = np.array([q[1] for q in x])
        r = np.array([q[2] for q in x])
        eq = np.cumsum(r[np.argsort([q[0] for q in x])])
        return {"trades": len(r), "reussite": round(float((r > 0).mean()), 3), "gain_moyen_R": round(float(r.mean()), 3),
                "gain_moyen_pb": round(float(p.mean() * 1e4), 2),
                "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else 0,
                "total_R": round(float(r.sum()), 1), "pire_baisse_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}
    res = []
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            for q in trades(a, s, **v):
                (te if q[0] >= cut else tr).append(q)
        res.append({"variante": v, "apprentissage": st(tr), "test": st(te)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    ok = [x for x in res if x["apprentissage"].get("trades", 0) >= 60 and x["apprentissage"].get("t", 0) >= 2]
    print("VARIANTES t >= 2 sur l'apprentissage :", len(ok), "/", len(res), flush=True)
    for x in sorted(ok, key=lambda x: -x["apprentissage"]["t"])[:6]:
        print("   RETENUE", json.dumps(x["variante"]), "| APPRENTISSAGE", json.dumps(x["apprentissage"]), "| TEST",
              json.dumps(x["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/fx_annonces.json", "w") as f:
        json.dump({"genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "15M")
