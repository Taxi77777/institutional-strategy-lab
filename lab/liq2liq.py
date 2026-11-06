"""Modele ICT « liquidite -> liquidite » (demande du trader, 6 nov.) : le prix va d'une zone de liquidite a l'autre.

Regles (fixes, pas d'optimisation fine : peu de risque de sur-apprentissage) :
1. Zones : haut / bas de la session asiatique (0 h - 7 h UTC), de Londres (7 h - 12 h UTC), de la veille, de la
   semaine precedente. Une zone n'est connue qu'une fois sa periode terminee (jamais de donnee future).
2. Declencheur, pendant Londres (7 h - 11 h UTC) ou New York (12 h - 16 h UTC) : le prix PREND une zone (meche sous
   un bas / au-dessus d'un haut : chasse des stops), puis CASSE LA STRUCTURE dans l'autre sens (cloture au-dela du
   dernier sommet / creux, avec une bougie d'impulsion >= 0,8 ATR) dans les 8 bougies.
3. Entree : ordre limite au milieu (ou au bord) du FVG laisse par l'impulsion, valable 8 bougies.
4. Stop : derriere la meche qui a pris la liquidite (+ 0,1 ATR).
5. Objectif : la prochaine zone de liquidite EN FACE (au moins 1 fois le risque, ou 2 selon la variante) ; moitie
   encaissee a 1 fois le risque puis stop a l'entree ; sortie forcee a 20 h UTC ou apres 64 bougies.
Le test : 60 % les plus anciens pour choisir la variante, 40 % les plus recents (jamais vus) pour le verdict.

   python -m lab.liq2liq fx 15M   -> rapport (reussite, gain moyen en R, signaux par jour) + resultats/liq2liq_fx.json"""
import json
import logging
import os
import sys
import time

import numpy as np

log = logging.getLogger("lab")
COST = {"fx": 0.00005, "cr": 0.0012}           # aller-retour en fraction du prix (spread forex / frais crypto)


def _atr(a, n=14):
    h, l, c = a[:, 2], a[:, 3], a[:, 4]
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    k = np.ones(n) / n
    out = np.convolve(tr, k, "full")[:len(tr)]
    out[:n] = np.nan
    return out


def pools(a):
    """Niveaux de liquidite connus a chaque bougie (NaN tant que la periode n'est pas terminee)."""
    ts, h, l = a[:, 0], a[:, 2], a[:, 3]
    n = len(a)
    day = (ts // 86400).astype(int)
    hr = ((ts % 86400) // 3600).astype(int)
    week = ((ts - 4 * 86400) // (7 * 86400)).astype(int)        # semaines du lundi au dimanche
    P = {k: np.full(n, np.nan) for k in ("ash", "asl", "loh", "lol", "pdh", "pdl", "pwh", "pwl")}
    # haut / bas par jour et par session
    dh, dl, ah, al, lh, ll = {}, {}, {}, {}, {}, {}
    for i in range(n):
        d = day[i]
        dh[d] = max(dh.get(d, -np.inf), h[i])
        dl[d] = min(dl.get(d, np.inf), l[i])
        if hr[i] < 7:
            ah[d] = max(ah.get(d, -np.inf), h[i])
            al[d] = min(al.get(d, np.inf), l[i])
        elif hr[i] < 12:
            lh[d] = max(lh.get(d, -np.inf), h[i])
            ll[d] = min(ll.get(d, np.inf), l[i])
    wh, wl = {}, {}
    for i in range(n):
        w = week[i]
        wh[w] = max(wh.get(w, -np.inf), h[i])
        wl[w] = min(wl.get(w, np.inf), l[i])
    prev_day = {}
    days = sorted(dh)
    for k, d in enumerate(days):
        prev_day[d] = days[k - 1] if k else None
    for i in range(n):
        d = day[i]
        pd_ = prev_day.get(d)
        if pd_ is not None:
            P["pdh"][i], P["pdl"][i] = dh[pd_], dl[pd_]
        if hr[i] >= 7 and d in ah:
            P["ash"][i], P["asl"][i] = ah[d], al[d]
        if hr[i] >= 12 and d in lh:
            P["loh"][i], P["lol"][i] = lh[d], ll[d]
        if week[i] - 1 in wh:
            P["pwh"][i], P["pwl"][i] = wh[week[i] - 1], wl[week[i] - 1]
    return P


NOMS = {"ash": "haut asiatique", "asl": "bas asiatique", "loh": "haut de Londres", "lol": "bas de Londres",
        "pdh": "haut de la veille", "pdl": "bas de la veille", "pwh": "haut de la semaine", "pwl": "bas de la semaine"}
HAUTS, BAS = ("ash", "loh", "pdh", "pwh"), ("asl", "lol", "pdl", "pwl")


def setups(a, P, atr, entry_mode="ce", min_rr=1.0, killzone=True):
    """Detecte les setups (sans les jouer) : liste de dict {i_signal, dir, entry, stop, target, cible, prise}."""
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((ts % 86400) // 3600).astype(int)
    day = (ts // 86400).astype(int)
    out = []
    taken = set()                                   # une zone prise une seule fois par jour
    n = len(a)
    for i in range(10, n - 3):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if killzone and not (7 <= hr[i] < 11 or 12 <= hr[i] < 16):
            continue
        for side, names in ((1, BAS), (-1, HAUTS)):          # 1 : liquidite sous le prix prise -> achat
            for nm in names:
                lv = P[nm][i]
                if not np.isfinite(lv) or (day[i], nm) in taken:
                    continue
                swept = (l[i] < lv and min(l[i - 3:i]) >= lv) if side == 1 else (h[i] > lv and max(h[i - 3:i]) <= lv)
                if not swept:
                    continue
                taken.add((day[i], nm))
                ref = max(h[i - 6:i + 1]) if side == 1 else min(l[i - 6:i + 1])     # structure a casser
                ext = l[i] if side == 1 else h[i]
                for j in range(i + 1, min(i + 9, n - 2)):
                    ext = min(ext, l[j]) if side == 1 else max(ext, h[j])
                    body = abs(c[j] - o[j])
                    mss = (c[j] > ref) if side == 1 else (c[j] < ref)
                    if not (mss and body >= 0.8 * atr[j]):
                        continue
                    k = j + 1                                   # FVG : bougies j-1 et j+1 ne se touchent pas
                    if side == 1 and l[k] > h[j - 1]:
                        lo_, hi_ = h[j - 1], l[k]
                    elif side == -1 and h[k] < l[j - 1]:
                        lo_, hi_ = h[k], l[j - 1]
                    else:
                        break
                    entry = (lo_ + hi_) / 2 if entry_mode == "ce" else (hi_ if side == 1 else lo_)
                    stop = ext - 0.1 * atr[j] if side == 1 else ext + 0.1 * atr[j]
                    risk = (entry - stop) * side
                    if risk <= 0:
                        break
                    tg = [(P[t][k], t) for t in (HAUTS if side == 1 else BAS) if np.isfinite(P[t][k])
                          and (P[t][k] - entry) * side >= min_rr * risk]
                    if not tg:
                        break
                    target, cible = min(tg, key=lambda x: (x[0] - entry) * side)
                    out.append({"i": k, "dir": side, "entry": float(entry), "stop": float(stop),
                                "target": float(target), "cible": cible, "prise": nm, "atr": float(atr[j])})
                    break
    return out


def play(a, s, kind, wait=8, max_bars=64):
    """Joue un setup : (ouvert a l'indice, R final, objectif 1 atteint, fermeture a l'indice) ou None si non rempli."""
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    d, e, sl, tg = s["dir"], s["entry"], s["stop"], s["target"]
    risk = (e - sl) * d
    tp1 = e + d * risk
    n = len(a)
    fill = None
    for k in range(s["i"] + 1, min(s["i"] + 1 + wait, n)):
        if (h[k] - tg) * d >= 0 and fill is None and ((l[k] > e) if d == 1 else (h[k] < e)):
            return None                             # objectif atteint avant le remplissage : setup annule
        if (l[k] <= e) if d == 1 else (h[k] >= e):
            fill = k
            break
    if fill is None:
        return None
    cost = COST[kind] * e / risk
    half, stop = False, sl
    for k in range(fill, min(fill + max_bars, n)):
        hr = int((ts[k] % 86400) // 3600)
        if (l[k] <= stop) if d == 1 else (h[k] >= stop):
            r = (0.5 + 0.5 * (stop - e) * d / risk) if half else (stop - e) * d / risk
            return fill, r - cost, half, k
        if not half and ((h[k] >= tp1) if d == 1 else (l[k] <= tp1)):
            half, stop = True, e
        if (h[k] >= tg) if d == 1 else (l[k] <= tg):
            r = 0.5 + 0.5 * (tg - e) * d / risk if half else (tg - e) * d / risk
            return fill, r - cost, True, k
        if hr >= 20 and k > fill:
            r = (0.5 + 0.5 * (c[k] - e) * d / risk) if half else (c[k] - e) * d / risk
            return fill, r - cost, half, k
    k = min(fill + max_bars, n) - 1
    r = (0.5 + 0.5 * (c[k] - e) * d / risk) if half else (c[k] - e) * d / risk
    return fill, r - cost, half, k


VARIANTES = [{"entry_mode": m, "min_rr": rr, "killzone": kz} for m in ("ce", "bord") for rr in (1.0, 2.0)
             for kz in (True, False)]


def run(kind, tf="15M", syms=None):
    from lab import data
    from lab.config import UNIVERSE
    syms = syms or UNIVERSE[kind]
    bars = {}
    for s in syms:
        try:
            a = data.long_bars(s, kind, tf)
            if len(a) > 2000:
                bars[s] = a
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([b[:, 0] for b in bars.values()])
    cut = np.quantile(t_all, 0.6)                    # 60 % anciens : choix ; 40 % recents : verdict
    days_test = (t_all.max() - cut) / 86400
    days_train = (cut - t_all.min()) / 86400
    res = []
    for v in VARIANTES:
        tr, te = [], []
        per = {}
        for s, a in bars.items():
            atr = _atr(a)
            P = pools(a)
            busy = -1
            for st in setups(a, P, atr, **v):
                if st["i"] <= busy:
                    continue                        # un trade a la fois par actif
                p = play(a, st, kind)
                if p is None:
                    continue
                fill, r, win, end = p
                busy = end
                (te if a[fill, 0] >= cut else tr).append((r, win))
                if a[fill, 0] >= cut:
                    per.setdefault(s, []).append(r)
        def stats(x, days):
            if not x:
                return {"trades": 0}
            r = np.array([y[0] for y in x])
            return {"trades": len(x), "reussite": round(float(np.mean([y[1] for y in x])), 3),
                    "gain_moyen_R": round(float(r.mean()), 3), "total_R": round(float(r.sum()), 1),
                    "par_jour": round(len(x) / max(1, days), 2)}
        res.append({"variante": v, "apprentissage": stats(tr, days_train), "test": stats(te, days_test),
                    "par_actif_test": {s: {"trades": len(x), "gain_R": round(float(np.sum(x)), 1)} for s, x in per.items()}})
    best = max(res, key=lambda x: x["apprentissage"].get("gain_moyen_R", -9) if x["apprentissage"].get("trades", 0) >= 50
               else -9)
    out = {"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "actifs": len(bars), "jours_test": round(days_test), "choisie": best["variante"], "variantes": res}
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/liq2liq_%s_%s.json" % (kind, tf), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for x in res:
        print("VARIANTE", json.dumps(x["variante"]), "| APPRENTISSAGE", json.dumps(x["apprentissage"]),
              "| TEST", json.dumps(x["test"]), flush=True)
    print("CHOISIE (sur l'apprentissage seulement)", json.dumps(best["variante"]), "-> TEST (jamais vu)",
          json.dumps(best["test"]), flush=True)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "15M")
