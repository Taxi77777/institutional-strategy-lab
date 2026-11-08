"""Modele « profil de volume + liquidite » (demande du trader, 8 nov.).
Le profil de volume de la veille (volume reparti sur les prix, ticks Dukascopy pour le forex, volume reel pour la
crypto) donne le POC (prix le plus traite), la zone de valeur (70 % du volume : VAH en haut, VAL en bas) et les bords
de la journee (plus haut / plus bas de la veille = liquidite : les stops sont juste au-dela).
  - « 80 » (regle des 80 % de Dalton) : la journee ouvre hors de la zone de valeur, puis le prix y revient et y reste
    (1 ou 2 clotures) -> il traverserait toute la zone : on joue vers l'autre bord (ou vers le POC) ;
  - « balayage » (liquidite) : le prix chasse les stops au-dela du plus haut / plus bas de la veille, puis revient
    dans la zone de valeur -> faux depart des banques : on joue vers le POC ou l'autre bord.
Entree a l'ouverture suivante ; stop au-dela de l'extreme de l'excursion (+0,1 ATR) ; moitie au premier objectif
puis stop a l'entree ; objectif au moins 1 fois le risque. Regle stricte (objectif jamais compte sur la bougie
d'entree). Test honnete : 60 % anciens pour choisir, 40 % recents (jamais vus) pour le verdict.
   python -m lab.vprofil fx 15M"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import _atr
from lab.ob_retour import play

log = logging.getLogger("lab")


def profile(h, l, v, bins=50):
    lo, hi = float(l.min()), float(h.max())
    if hi <= lo:
        return None
    w = (hi - lo) / bins
    vol = np.zeros(bins)
    for hh, ll, vv in zip(h, l, v):
        a, b = int((ll - lo) / w), min(bins - 1, int((hh - lo) / w))
        vol[a:b + 1] += vv / (b - a + 1)
    if vol.sum() <= 0:
        return None
    p = int(vol.argmax())
    il, iu, s = p, p, vol[p]
    while s < 0.7 * vol.sum():
        up = vol[iu + 1] if iu + 1 < bins else -1
        dn = vol[il - 1] if il > 0 else -1
        if up >= dn:
            iu += 1
            s += up
        else:
            il -= 1
            s += dn
    return {"poc": lo + (p + 0.5) * w, "vah": lo + (iu + 1) * w, "val": lo + il * w, "pdh": hi, "pdl": lo}


def setups(a, vol, atr, model="80", conf=1, cible="bord", kz=True):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    day = (ts // 86400).astype(int)
    hr = ((ts % 86400) // 3600).astype(int)
    idx = {}
    for i, d in enumerate(day):
        idx.setdefault(d, []).append(i)
    days = sorted(idx)
    out = []
    for k in range(1, len(days)):
        prev, cur = idx[days[k - 1]], idx[days[k]]
        if len(prev) < 8 or len(cur) < 4:
            continue
        P = profile(h[prev], l[prev], vol[prev])
        if P is None:
            continue
        i0 = cur[0]
        for d in (1, -1):                                    # 1 : retour par le bas -> achat ; -1 : par le haut
            edge_out = P["val"] if d == 1 else P["vah"]
            if model == "80" and not ((o[i0] < P["val"]) if d == 1 else (o[i0] > P["vah"])):
                continue
            ext, swept, n_in = (l[i0] if d == 1 else h[i0]), False, 0
            for i in cur[:-1]:
                ext = min(ext, l[i]) if d == 1 else max(ext, h[i])
                if (l[i] < P["pdl"]) if d == 1 else (h[i] > P["pdh"]):
                    swept = True
                outside = (ext < edge_out) if d == 1 else (ext > edge_out)
                inside = P["val"] < c[i] < P["vah"]
                n_in = n_in + 1 if (outside and inside) else 0
                if n_in < conf or (model == "balayage" and not swept):
                    continue
                if kz and not 7 <= hr[i] < 17:
                    break
                if not np.isfinite(atr[i]) or atr[i] <= 0:
                    break
                e = o[i + 1]
                sl = ext - d * 0.1 * atr[i]
                tg = P["poc"] if cible == "poc" else (P["vah"] if d == 1 else P["val"])
                risk = (e - sl) * d
                if risk >= 0.2 * atr[i] and (tg - e) * d >= risk:
                    out.append({"i": i + 1, "dir": d, "entry": float(e), "stop": float(sl), "target": float(tg)})
                break                                         # un seul essai par cote et par jour
    out.sort(key=lambda x: x["i"])
    return out


VARIANTES = [{"model": m, "conf": n, "cible": t, "tp1": p} for m in ("80", "balayage") for n in (1, 2)
             for t in ("bord", "poc") for p in (0.5, 1.0)]


def run(kind, tf="15M"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, tf)
            if len(a) <= 1500:
                continue
            v = a[:, 5].copy()
            if kind == "fx" or v.sum() <= 0:                  # forex : volume des ticks garde a part par data
                vt = getattr(data, "VOLUME", {}).get(s)
                m = dict(zip(vt[0].astype(int), vt[1])) if vt is not None else {}
                v = np.array([m.get(int(t), np.nan) for t in a[:, 0]])
                if not np.isfinite(v).any() or np.nansum(v) <= 0:
                    v = np.ones(len(a))
                    log.warning("%s : pas de volume, profil en temps passe a chaque prix", s)
                v = np.where(np.isfinite(v), v, np.nanmedian(v))
            bars[s] = (a, v, _atr(a))
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    print("ACTIFS", len(bars), "avec volume reel :", sum(1 for a, v, _ in bars.values() if np.ptp(v) > 0), flush=True)
    t_all = np.concatenate([b[0][:, 0] for b in bars.values()])
    cut = np.quantile(t_all, 0.6)
    dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
    cache, res = {}, []
    for v in VARIANTES:
        tr, te = [], []
        for s, (a, vol, atr) in bars.items():
            k = (s, v["model"], v["conf"], v["cible"])
            if k not in cache:
                cache[k] = setups(a, vol, atr, v["model"], v["conf"], v["cible"], kz=(kind == "fx"))
            busy = -1
            for st in cache[k]:
                if st["i"] <= busy:
                    continue
                p = play(a, st, kind, tp1_r=v["tp1"])
                if p is None:
                    continue
                r, win, end = p
                busy = end
                (te if a[st["i"], 0] >= cut else tr).append((r, win))

        def stats(x, days):
            if not x:
                return {"trades": 0}
            r = np.array([y[0] for y in x])
            return {"trades": len(x), "reussite": round(float(np.mean([y[1] for y in x])), 3),
                    "gain_moyen_R": round(float(r.mean()), 3), "total_R": round(float(r.sum()), 1),
                    "par_jour": round(len(x) / max(1, days), 2)}
        res.append({"variante": v, "apprentissage": stats(tr, dtr), "test": stats(te, dte)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    best = max(res, key=lambda x: x["apprentissage"].get("gain_moyen_R", -9)
               if x["apprentissage"].get("trades", 0) >= 50 else -9)
    print("CHOISIE (sur l'apprentissage seulement)", json.dumps(best["variante"]), "-> TEST (jamais vu)",
          json.dumps(best["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/vprofil_%s_%s.json" % (kind, tf), "w") as f:
        json.dump({"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "actifs": len(bars), "choisie": best["variante"], "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "15M")
