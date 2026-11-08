"""Modele « Market Profile en TPO (lettres) » (demande du trader, 8 nov.). Chaque tranche de 30 minutes recoit une
lettre (A, B, C...) ; le profil compte, pour chaque prix, combien de lettres l'ont touche. Lecture classique
(Steidlmayer, Dalton) sur le profil de la veille et sur l'Initial Balance du jour :
  - « exces » : au plus haut (bas) de la veille, une queue de prix touches par une seule lettre = rejet fort ;
    au retour du prix dans la queue, on vend sous le plus haut (achete au-dessus du plus bas), objectif le POC TPO ;
  - « poor » : plus haut (bas) de la veille plat, touche par plusieurs lettres = enchere inachevee, de la liquidite
    reste a prendre ; quand le prix repasse au-dessus (dessous) du POC, on joue vers ce plus haut (bas) ;
  - « ib_faux » : l'Initial Balance (2 ou 4 premieres lettres de la seance : Londres en forex, 0 h UTC en crypto)
    est cassee puis le prix revient dedans -> faux depart, objectif l'autre bord ;
  - « ib_ext » : cloture au-dela de l'Initial Balance -> continuation, objectif 1 fois (ou 0,5 fois) sa hauteur.
Moitie au premier objectif puis stop a l'entree ; regle stricte (objectif jamais compte sur la bougie d'entree).
Test honnete : 60 % anciens pour choisir, 40 % recents (jamais vus) pour le verdict.   python -m lab.tpo fx"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import _atr
from lab.ob_retour import play

log = logging.getLogger("lab")
BINS = 40


def tpo(h, l):
    """Profil TPO d'une journee de bougies 30 min."""
    lo, hi = float(l.min()), float(h.max())
    if hi <= lo:
        return None
    w = (hi - lo) / BINS
    cnt = np.zeros(BINS, dtype=int)
    for hh, ll in zip(h, l):
        cnt[int((ll - lo) / w):min(BINS - 1, int((hh - lo) / w)) + 1] += 1
    p = int(np.argmax(cnt))
    il, iu, s, tot = p, p, cnt[p], cnt.sum()
    while s < 0.7 * tot:
        up = cnt[iu + 1] if iu + 1 < BINS else -1
        dn = cnt[il - 1] if il > 0 else -1
        if up >= dn:
            iu += 1
            s += up
        else:
            il -= 1
            s += dn
    tail_top = next((k for k in range(BINS) if cnt[BINS - 1 - k] > 1), BINS)
    tail_bot = next((k for k in range(BINS) if cnt[k] > 1), BINS)
    return {"lo": lo, "hi": hi, "w": w, "poc": lo + (p + 0.5) * w, "vah": lo + (iu + 1) * w, "val": lo + il * w,
            "top": int(cnt[-1]), "bot": int(cnt[0]), "tail_top": tail_top, "tail_bot": tail_bot}


def setups(a, atr, model, param, kz=True):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    day = (ts // 86400).astype(int)
    hr = (ts % 86400) / 3600.0
    idx = {}
    for i, d in enumerate(day):
        idx.setdefault(d, []).append(i)
    days = sorted(idx)
    out = []
    ib_start = 7.0 if kz else 0.0
    for k in range(1, len(days)):
        prev, cur = idx[days[k - 1]], idx[days[k]]
        if len(prev) < 16 or len(cur) < 8:
            continue
        P = tpo(h[prev], l[prev])
        if P is None:
            continue
        ok = [i for i in cur[:-1] if (not kz or 7 <= hr[i] < 17) and np.isfinite(atr[i]) and atr[i] > 0]
        if model == "exces":
            for d in (-1, 1):                                  # -1 : queue en haut -> vente ; 1 : en bas -> achat
                tail = P["tail_top"] if d == -1 else P["tail_bot"]
                if tail < param or tail >= BINS:
                    continue
                edge = P["hi"] if d == -1 else P["lo"]
                e = edge + d * tail * P["w"]                   # entree au debut de la queue
                for i in ok:
                    if (h[i] >= e) if d == -1 else (l[i] <= e):
                        if (o[i] - edge) * d < 0:              # ouverture deja au-dela du plus haut / bas : non
                            break
                        sl = edge - d * 0.1 * atr[i]
                        risk = (e - sl) * d
                        if risk >= 0.2 * atr[i] and (P["poc"] - e) * d >= risk:
                            out.append({"i": i, "dir": d, "entry": float(e), "stop": float(sl), "target": P["poc"]})
                        break
        elif model == "poor":
            for d in (1, -1):                                  # 1 : plus haut plat -> on va le chercher
                if (P["top"] if d == 1 else P["bot"]) < param:
                    continue
                edge = P["hi"] if d == 1 else P["lo"]
                for j, i in enumerate(ok):
                    if (h[i] >= edge) if d == 1 else (l[i] <= edge):
                        break                                  # deja pris
                    if (c[i] - P["poc"]) * d > 0:
                        e = o[i + 1]
                        win = ok[max(0, j - 3):j + 1]
                        sl = (min(l[x] for x in win) if d == 1 else max(h[x] for x in win)) - d * 0.1 * atr[i]
                        tg = edge + d * 0.1 * atr[i]
                        risk = (e - sl) * d
                        if risk >= 0.2 * atr[i] and (tg - e) * d >= risk:
                            out.append({"i": i + 1, "dir": d, "entry": float(e), "stop": float(sl), "target": float(tg)})
                        break
        else:                                                  # Initial Balance de la seance
            n_ib = int(param) if model == "ib_faux" else 2
            ib = [i for i in cur if ib_start <= hr[i] < ib_start + n_ib / 2.0]
            if len(ib) < n_ib:
                continue
            ibh, ibl = h[ib].max(), l[ib].min()
            rng = ibh - ibl
            after = [i for i in ok if hr[i] >= ib_start + n_ib / 2.0]
            for d in (1, -1):                                  # 1 : cassure du bas (faux) / du haut (extension)
                ext, broke = None, False
                for i in after:
                    if model == "ib_faux":
                        if (l[i] < ibl) if d == 1 else (h[i] > ibh):
                            broke = True
                        if broke:
                            ext = (min(ext, l[i]) if ext is not None else l[i]) if d == 1 else \
                                  (max(ext, h[i]) if ext is not None else h[i])
                            if ibl < c[i] < ibh:
                                e = o[i + 1]
                                sl = ext - d * 0.1 * atr[i]
                                tg = ibh if d == 1 else ibl
                                risk = (e - sl) * d
                                if risk >= 0.2 * atr[i] and (tg - e) * d >= risk:
                                    out.append({"i": i + 1, "dir": d, "entry": float(e), "stop": float(sl),
                                                "target": float(tg)})
                                break
                    else:
                        if (c[i] > ibh) if d == 1 else (c[i] < ibl):
                            e = o[i + 1]
                            sl = (ibh + ibl) / 2
                            tg = (ibh if d == 1 else ibl) + d * param * rng
                            risk = (e - sl) * d
                            if risk >= 0.2 * atr[i] and (tg - e) * d >= risk:
                                out.append({"i": i + 1, "dir": d, "entry": float(e), "stop": float(sl),
                                            "target": float(tg)})
                            break
    out.sort(key=lambda x: x["i"])
    return out


VARIANTES = ([{"model": "exces", "param": p, "tp1": t} for p in (2, 4) for t in (0.5, 1.0)] +
             [{"model": "poor", "param": p, "tp1": t} for p in (2, 3) for t in (0.5, 1.0)] +
             [{"model": "ib_faux", "param": p, "tp1": t} for p in (2, 4) for t in (0.5, 1.0)] +
             [{"model": "ib_ext", "param": p, "tp1": t} for p in (0.5, 1.0) for t in (0.5, 1.0)])


def run(kind, tf="30M"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, "30M")
            if len(a) > 1500:
                bars[s] = (a, _atr(a))
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    print("ACTIFS", len(bars), flush=True)
    t_all = np.concatenate([b[0][:, 0] for b in bars.values()])
    cut = np.quantile(t_all, 0.6)
    dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
    cache, res = {}, []
    for v in VARIANTES:
        tr, te = [], []
        for s, (a, atr) in bars.items():
            k = (s, v["model"], v["param"])
            if k not in cache:
                cache[k] = setups(a, atr, v["model"], v["param"], kz=(kind == "fx"))
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
    with open("resultats/tpo_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "actifs": len(bars), "choisie": best["variante"], "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1])
