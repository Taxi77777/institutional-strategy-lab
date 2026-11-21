"""AMD + retest du niveau de cassure 20 jours par la manipulation (observation du trader sur USDMXN et GBPUSD, 21 nov.).
Signal : cassure 20 jours dans le sens de la cloture d'il y a 90 jours (comme les indicateurs MT4).
Sessions (heures UTC) :
  « londres »  : Asie 0-7 h, manipulation 7-12 h, entree de secours 16 h (indicateurs actuels) ;
  « newyork »  : accumulation 0-12 h, manipulation 12-15 h, entree de secours 17 h (USDMXN, USDCAD...).
Gestions :
  « swing » (TENDANCE-AMD-90) : stop 4 ATR, moitie a 0,5 ATR puis stop a l'entree, stop suiveur journalier ;
  « h24 »   (AMD-H24)         : stop 1 ATR, moitie a 0,2 ATR puis stop a l'entree, reste a 1 ATR, sans heure de sortie.
Frais : 0,005 % forex, 0,03 % USDMXN, 0,02 % or / argent. 60 % anciens / 40 % recents, resultats par paire.
python -m lab.amd_par_paire fx 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.tendance_amd import daily

log = logging.getLogger("lab")
PAIRES = ("EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,EURGBP,EURJPY,EURCHF,EURCAD,EURAUD,EURNZD,GBPJPY,GBPCHF,"
          "GBPCAD,GBPAUD,GBPNZD,AUDJPY,CADJPY,CHFJPY,NZDJPY,AUDCAD,AUDCHF,AUDNZD,CADCHF,NZDCAD,NZDCHF,USDMXN,XAUUSD,XAGUSD").split(",")
SESSIONS = {"londres": (7, 12, 16), "newyork": (12, 15, 17)}


def cost(sym):
    return 3e-4 if sym == "USDMXN" else (2e-4 if sym.startswith("XA") else 5e-5)


def trades(a, sym, sess="londres", gestion="swing", n=20, fond=90, retest=None, kind="fx"):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr = daily(a)
    is_close = np.zeros(len(a), bool)
    is_close[en] = True
    a_end, m_end, fb = SESSIONS[sess]
    k, t1, tp2 = (4.0, 0.5, None) if gestion == "swing" else (1.0, 0.2, 1.0)
    out, i = [], max(n, fond, 21) + 1
    while i < len(D) - 2:
        d = 0
        if np.isfinite(atr[i]) and atr[i] > 0:
            if D[i, 4] > D[i - n:i, 2].max() and D[i, 4] > D[i - fond, 4]:
                d = 1
            elif D[i, 4] < D[i - n:i, 3].min() and D[i, 4] < D[i - fond, 4]:
                d = -1
        if not d:
            i += 1
            continue
        A = atr[i]
        lvl = D[i - n:i, 2].max() if d == 1 else D[i - n:i, 3].min()     # niveau casse (ancien plus haut / plus bas 20 j)
        rows = range(st[i + 1], en[i + 1] + 1)
        acc = [r for r in rows if hr[r] < a_end]
        m = None
        amd = False
        ext = None
        if len(acc) >= 4:
            al, ah = l[acc].min(), h[acc].max()
            manip = False
            for r in rows:
                if hr[r] < a_end:
                    continue
                if not manip and hr[r] < m_end and ((l[r] < al) if d == 1 else (h[r] > ah)):
                    manip = True
                if manip:
                    w_ = l[r] if d == 1 else h[r]
                    ext = w_ if ext is None else (min(ext, w_) if d == 1 else max(ext, w_))
                if manip and ((c[r] > al) if d == 1 else (c[r] < ah)) and r + 1 < len(a):
                    m, amd = r + 1, True
                    break
                if hr[r] >= fb:
                    break
        if retest is not None:
            # garder seulement les manipulations qui reviennent toucher le niveau casse (a tol ATR pres) sans le traverser loin
            tol, plancher = retest
            if not amd or ext is None:
                i += 1
                continue
            dist = (ext - lvl) * d / A          # > 0 : la meche est restee du bon cote du niveau ; < 0 : elle l'a traverse
            if not (dist <= tol and dist >= -plancher):
                i += 1
                continue
        if m is None:
            m = next((r for r in rows if hr[r] >= fb), None)
        if m is None or m >= len(a) - 1:
            i += 1
            continue
        e = o[m]
        risk = k * A
        stop, tg1, half, best = e - d * risk, e + d * t1 * A, False, e
        tg2 = e + d * tp2 * A if tp2 else None
        x, j = None, m
        while j < len(a):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x = stop if j == m else (min(o[j], stop) if d == 1 else max(o[j], stop))
                break
            if not half and ((h[j] >= tg1) if d == 1 else (l[j] <= tg1)):
                half = True
                stop = max(stop, e) if d == 1 else min(stop, e)
            if tg2 is not None and ((h[j] >= tg2) if d == 1 else (l[j] <= tg2)):
                x = tg2
                break
            if gestion == "swing" and is_close[j]:
                best = max(best, c[j]) if d == 1 else min(best, c[j])
                stop = max(stop, best - risk) if d == 1 else min(stop, best + risk)
            j += 1
        if x is None:
            break
        r = (0.5 * t1 * A / risk + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
        out.append((a[m, 0], a[j, 0], d, float(r - (cost(sym) if kind == "fx" else 0.0012) * e / risk), amd))
        i = int(np.searchsorted(D[:, 0], a[j, 0] // 86400)) + 1
    return out


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    syms = PAIRES if kind == "fx" else UNIVERSE["cr"]
    bars = {}
    for s in syms:
        try:
            a = data.long_bars(s, kind, "1H")
            if kind == "fx":
                a = a[((a[:, 0] // 86400 + 3) % 7) < 5]
            if len(a) > 24 * 300:
                bars[s] = a[:, :5]
        except Exception as ex:
            print("ABSENT", s, ex, flush=True)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = float(np.quantile(t_all, 0.6))
    print("ACTIFS", kind, len(bars), "coupure", time.strftime("%d/%m/%Y", time.gmtime(cut)), flush=True)

    def st(x):
        if not x:
            return {"trades": 0}
        r = np.array([t[3] for t in x])
        return {"trades": len(r), "reussite": round(float((r > 0).mean()), 3), "gain_moyen_R": round(float(r.mean()), 3),
                "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else 0,
                "total_R": round(float(r.sum()), 1)}
    res = []
    variantes = [None, (0.1, 0.5), (0.25, 0.5), (0.5, 0.5), (0.25, 1.0), (0.5, 1.5)]
    for g in ("swing", "h24"):
        for v in variantes:
            tr, te = [], []
            for s, a in bars.items():
                for t in trades(a, s, "londres", g, retest=v, kind=kind):
                    (te if t[0] >= cut else tr).append(t)
            nm = "%s | %s" % (g, "BASE (sans filtre)" if v is None else "retest niveau 20 j : tol %.2f ATR, traverse max %.1f ATR" % v)
            res.append({"variante": nm, "apprentissage": st(tr), "test": st(te)})
            print("VARIANTE", nm, "| APPRENTISSAGE", json.dumps(st(tr)), "| TEST", json.dumps(st(te)), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_retest_%s.json" % kind, "w") as f:
        json.dump({"genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "resultats": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
