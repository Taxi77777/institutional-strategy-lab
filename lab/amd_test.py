"""Modele AMD (Accumulation - Manipulation - Distribution, « Power of 3 » ICT), demande du trader (15 nov.).
  A. Accumulation : la fourchette de la session asiatique (0 h - 7 h UTC). La liquidite (stops) s'accumule juste
     au-dessus de son plus haut et juste sous son plus bas.
  M. Manipulation : pendant Londres (7 h - 10 h UTC, ou jusqu'a 13 h), le prix chasse UN cote de la fourchette
     (faux depart / Judas swing) : les stops sautent, les banques se remplissent.
  D. Distribution : le prix revient (cloture de nouveau dans la fourchette, ou au-dela de l'ouverture du jour a 0 h
     UTC) -> on joue l'autre sens, vers la liquidite d'en face (l'autre bord de l'Asie, ou au-dela).
Stop : extreme de la manipulation + 0,1 ATR. Sortie au plus tard a 21 h UTC (fin de New York). Moitie au premier
objectif puis stop a l'entree. Entree a l'ouverture de la bougie suivante. Regle stricte : l'objectif n'est jamais
compte sur la bougie d'entree. Un seul trade par jour et par actif. Test honnete : 60 % anciens pour choisir la
variante, 40 % recents (jamais vus) pour le verdict.   python -m lab.amd_test fx 15M"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import _atr
from lab.ob_retour import play

log = logging.getLogger("lab")


def setups(a, atr, declencheur="retour", serre=0.0, cible="bord", fin_manip=10, adr=None):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    day = (ts // 86400).astype(int)
    hr = (ts % 86400) / 3600.0
    idx = {}
    for i, d in enumerate(day):
        idx.setdefault(d, []).append(i)
    days = sorted(idx)
    out = []
    for k, dd in enumerate(days):
        rows = idx[dd]
        asie = [i for i in rows if hr[i] < 7]
        if len(asie) < 4:
            continue
        ah, al = h[asie].max(), l[asie].min()
        rng = ah - al
        if rng <= 0:
            continue
        if serre and adr is not None:
            ref = adr.get(dd)
            if ref is None or rng > serre * ref:              # accumulation : fourchette etroite seulement
                continue
        op = o[rows[0]]                                        # ouverture du jour (0 h UTC)
        prev = idx.get(days[k - 1]) if k > 0 else None
        pdh = h[prev].max() if prev else np.nan
        pdl = l[prev].min() if prev else np.nan
        londres = [i for i in rows if 7 <= hr[i] < fin_manip]
        apres = [i for i in rows if 7 <= hr[i] < 20]
        end_i = max([i for i in rows if hr[i] < 21], default=rows[-1])
        side, ext = 0, None
        for i in apres:
            if side == 0 and i in londres:
                up, dn = h[i] > ah, l[i] < al
                if up and dn:
                    break                                      # les deux cotes pris : pas d'AMD propre
                if up:
                    side, ext = -1, h[i]                       # chasse en haut -> on vendra
                elif dn:
                    side, ext = 1, l[i]                        # chasse en bas -> on achetera
            if side == 0:
                continue
            ext = max(ext, h[i]) if side == -1 else min(ext, l[i])
            if (l[i] < al) if side == -1 else (h[i] > ah):
                break                                          # l'autre cote est pris aussi : invalide
            if declencheur == "retour":
                ok = al < c[i] < ah
            else:                                              # retour de l'autre cote de l'ouverture du jour
                ok = (c[i] < op) if side == -1 else (c[i] > op)
                ok = ok and al < c[i] < ah
            if not ok:
                continue
            if i + 1 > end_i or not np.isfinite(atr[i]) or atr[i] <= 0:
                break
            e = o[i + 1]
            d = side
            sl = ext + 0.1 * atr[i] if d == -1 else ext - 0.1 * atr[i]
            if cible == "bord":
                tg = al if d == -1 else ah
            elif cible == "veille":
                tg = pdl if d == -1 else pdh
                if not np.isfinite(tg) or (tg - e) * d <= 0:
                    tg = al if d == -1 else ah
            else:                                              # extension : une fourchette au-dela de l'autre bord
                tg = (al - rng) if d == -1 else (ah + rng)
            risk = (e - sl) * d
            if risk >= 0.2 * atr[i] and (tg - e) * d >= risk:
                out.append({"i": i + 1, "dir": d, "entry": float(e), "stop": float(sl), "target": float(tg),
                            "fin": int(end_i)})
            break
    return out


VARIANTES = [{"declencheur": dc, "serre": s, "cible": t, "fin_manip": f, "tp1": p}
             for dc in ("retour", "ouverture") for s in (0.0, 0.6) for t in ("bord", "veille", "extension")
             for f in (10,) for p in (0.5, 1.0)] + \
            [{"declencheur": "retour", "serre": 0.0, "cible": t, "fin_manip": 13, "tp1": p}
             for t in ("bord", "extension") for p in (0.5, 1.0)]


def run(kind, tf="15M"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, tf)
            if kind == "fx":
                a = a[((a[:, 0] // 86400 + 3) % 7) < 5]       # du lundi au vendredi
            if len(a) <= 1500:
                continue
            day = (a[:, 0] // 86400).astype(int)
            rngs = {}
            for d in np.unique(day):
                m = day == d
                rngs[d] = a[m, 2].max() - a[m, 3].min()
            ks = sorted(rngs)
            adr = {}
            for j, d in enumerate(ks):                         # amplitude moyenne des 20 jours precedents
                if j >= 20:
                    adr[d] = float(np.mean([rngs[x] for x in ks[j - 20:j]]))
            bars[s] = (a, _atr(a), adr)
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    print("ACTIFS", kind, tf, len(bars), flush=True)
    t_all = np.concatenate([b[0][:, 0] for b in bars.values()])
    cut = np.quantile(t_all, 0.6)
    dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
    cache, res = {}, []
    for v in VARIANTES:
        tr, te = [], []
        for s, (a, atr, adr) in bars.items():
            key = (s, v["declencheur"], v["serre"], v["cible"], v["fin_manip"])
            if key not in cache:
                cache[key] = setups(a, atr, v["declencheur"], v["serre"], v["cible"], v["fin_manip"], adr)
            for st in cache[key]:
                p = play(a, st, kind, max_bars=max(1, st["fin"] - st["i"] + 1), tp1_r=v["tp1"])
                if p is None:
                    continue
                r, win, end = p
                (te if a[st["i"], 0] >= cut else tr).append((r, win, st["dir"]))

        def stats(x, days):
            if not x:
                return {"trades": 0}
            r = np.array([y[0] for y in x])
            return {"trades": len(x), "reussite": round(float(np.mean([y[1] for y in x])), 3),
                    "gain_moyen_R": round(float(r.mean()), 3), "total_R": round(float(r.sum()), 1),
                    "par_jour": round(len(x) / max(1, days), 2),
                    "achats_R": round(float(sum(y[0] for y in x if y[2] == 1)), 1),
                    "ventes_R": round(float(sum(y[0] for y in x if y[2] == -1)), 1)}
        res.append({"variante": v, "apprentissage": stats(tr, dtr), "test": stats(te, dte)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    best = max(res, key=lambda x: x["apprentissage"].get("gain_moyen_R", -9)
               if x["apprentissage"].get("trades", 0) >= 50 else -9)
    print("CHOISIE (sur l'apprentissage seulement)", json.dumps(best["variante"]), "-> TEST (jamais vu)",
          json.dumps(best["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_test_%s_%s.json" % (kind, tf), "w") as f:
        json.dump({"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "actifs": len(bars), "choisie": best["variante"], "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "15M")
