"""Forex intraday : retour a la moyenne pendant la session asiatique (demande du trader, 16 nov. 22 h : intraday forex).
La nuit (21 h - 6 h UTC), peu de nouvelles et peu de volume : les paires calmes oscillent dans un couloir.
Bougies 15 min. A chaque bougie de la fenetre : ecart z = (cloture - moyenne des N dernieres bougies) / ecart-type ;
si |z| >= seuil, on joue le retour : entree a l'ouverture suivante, sortie quand le prix revient a la moyenne
(ou a la moitie du chemin), stop a s ecarts-types au-dela, fermeture forcee a 6 h UTC. Une position a la fois.
Frais de nuit plus eleves (spread large) : 2,5 points de base aller-retour. 60 % anciens pour choisir, 40 % recents.
python -m lab.fx_asie_mr fx 15M"""
import json
import logging
import os
import sys
import time

import numpy as np

log = logging.getLogger("lab")
COST = 2.5e-4
PAIRES_CALMES = ("EURGBP", "AUDNZD", "EURCHF", "USDCHF", "EURUSD", "USDCAD", "AUDUSD", "NZDUSD", "GBPUSD", "USDJPY",
                 "EURAUD", "EURCAD", "CADJPY", "AUDJPY", "GBPCHF", "CHFJPY")


def in_win(h, deb, fin):
    return (h >= deb) | (h < fin) if deb > fin else (h >= deb) & (h < fin)


def trades(a, N=20, z0=2.0, s=1.5, sortie="moyenne", deb=22, fin=6, tendance=False):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    cs, cs2 = np.cumsum(np.r_[0, c]), np.cumsum(np.r_[0, c * c])
    mu = np.full(len(c), np.nan)
    sd = np.full(len(c), np.nan)
    mu[N - 1:] = (cs[N:] - cs[:-N]) / N
    sd[N - 1:] = np.sqrt(np.maximum((cs2[N:] - cs2[:-N]) / N - mu[N - 1:] ** 2, 0))
    ema = None
    if tendance:                                            # filtre : pas de pari contre une forte pente de 4 h
        ema = np.full(len(c), np.nan)
        ema[16:] = c[16:] - c[:-16]
    win = in_win(hr, deb, fin)
    out, k = [], N
    while k < len(c) - 2:
        if not win[k] or not np.isfinite(sd[k]) or sd[k] <= 0:
            k += 1
            continue
        z = (c[k] - mu[k]) / sd[k]
        if abs(z) < z0:
            k += 1
            continue
        d = -1 if z > 0 else 1
        if ema is not None and np.isfinite(ema[k]) and abs(ema[k]) > 3 * sd[k] and np.sign(ema[k]) == -d:
            k += 1
            continue
        e = o[k + 1]
        tgt = mu[k] if sortie == "moyenne" else (e + mu[k]) / 2
        stop = e - d * s * sd[k]
        if (tgt - e) * d <= 0:
            k += 1
            continue
        x, j = None, k + 1
        while j < len(c):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x = stop if j == k + 1 else (min(o[j], stop) if d == 1 else max(o[j], stop))
                break
            if (h[j] >= tgt) if d == 1 else (l[j] <= tgt):
                x = tgt
                break
            if not win[j] or (a[j, 0] - a[k + 1, 0]) > 10 * 3600:
                x = o[j]
                break
            j += 1
        if x is None:
            break
        risk = s * sd[k]
        out.append((a[k + 1, 0], (x - e) * d / e - COST, (x - e) * d / risk - COST * e / risk))
        k = j + 1
    return out


VARIANTES = [{"N": N, "z0": z0, "s": s, "sortie": so, "tendance": tf}
             for N in (20, 48) for z0 in (1.5, 2.0, 2.5) for s in (1.0, 2.0) for so in ("moyenne", "moitie")
             for tf in (False, True)]


def run(kind, tf="15M"):
    from lab import data
    if kind != "fx":
        print("forex seulement")
        return
    bars = {}
    for s in PAIRES_CALMES:
        try:
            a = data.long_bars(s, "fx", "15M")
            a = a[((a[:, 0] // 86400 + 3) % 7) < 5]
            if len(a) > 96 * 200:
                bars[s] = a[:, :5]
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = np.quantile(t_all, 0.6)
    yrs_te = (t_all.max() - cut) / 86400 / 365
    print("ACTIFS", len(bars), "du", time.strftime("%d/%m/%Y", time.gmtime(t_all.min())), "coupure",
          time.strftime("%d/%m/%Y", time.gmtime(cut)), flush=True)

    def st(x, yrs):
        if not x:
            return {"trades": 0}
        p = np.array([t[1] for t in x])
        r = np.array([t[2] for t in x])
        o = np.argsort([t[0] for t in x])
        eq = np.cumsum(r[o])
        return {"trades": len(r), "par_jour": round(len(r) / yrs / 260, 2), "reussite": round(float((r > 0).mean()), 3),
                "gain_moyen_R": round(float(r.mean()), 3), "gain_moyen_pb": round(float(p.mean() * 1e4), 2),
                "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else 0,
                "R_par_an": round(float(r.sum() / yrs), 1), "pire_baisse_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}
    yrs_tr = (cut - t_all.min()) / 86400 / 365
    res, par_paire = [], {}
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            for t in trades(a, **v):
                (te if t[0] >= cut else tr).append(t)
        res.append({"variante": v, "apprentissage": st(tr, yrs_tr), "test": st(te, yrs_te)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    ok = [x for x in res if x["apprentissage"].get("trades", 0) >= 150 and x["apprentissage"].get("t", 0) >= 2]
    print("VARIANTES t >= 2 sur l'apprentissage :", len(ok), "/", len(res), flush=True)
    for x in sorted(ok, key=lambda x: -x["apprentissage"]["t"])[:6]:
        print("   RETENUE", json.dumps(x["variante"]), "| APPRENTISSAGE", json.dumps(x["apprentissage"]), "| TEST",
              json.dumps(x["test"]), flush=True)
    # par paire, pour la meilleure variante de l'apprentissage (choix des paires sur l'apprentissage seulement)
    if ok:
        v = max(ok, key=lambda x: x["apprentissage"]["t"])["variante"]
        keep_te = []
        for s, a in bars.items():
            tt = trades(a, **v)
            tr = [t for t in tt if t[0] < cut]
            te = [t for t in tt if t[0] >= cut]
            par_paire[s] = {"apprentissage": st(tr, yrs_tr), "test": st(te, yrs_te)}
            if st(tr, yrs_tr).get("t", 0) >= 2:
                keep_te += te
        print("PAIRES t >= 2 sur l'apprentissage -> TEST", json.dumps(st(keep_te, yrs_te)), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/fx_asie_mr.json", "w") as f:
        json.dump({"genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res, "par_paire": par_paire},
                  f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "15M")
