"""AMD paire par paire, sessions classiques ou decalees sur New York (demande du trader, 20 nov.).
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


def trades(a, sym, sess="londres", gestion="swing", n=20, fond=90):
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
        rows = range(st[i + 1], en[i + 1] + 1)
        acc = [r for r in rows if hr[r] < a_end]
        m = None
        amd = False
        if len(acc) >= 4:
            al, ah = l[acc].min(), h[acc].max()
            manip = False
            for r in rows:
                if hr[r] < a_end:
                    continue
                if not manip and hr[r] < m_end and ((l[r] < al) if d == 1 else (h[r] > ah)):
                    manip = True
                if manip and ((c[r] > al) if d == 1 else (c[r] < ah)) and r + 1 < len(a):
                    m, amd = r + 1, True
                    break
                if hr[r] >= fb:
                    break
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
        out.append((a[m, 0], a[j, 0], d, float(r - cost(sym) * e / risk), amd))
        i = int(np.searchsorted(D[:, 0], a[j, 0] // 86400)) + 1
    return out


def run(kind, tf="1H"):
    from lab import data
    if kind != "fx":
        print("forex seulement")
        return
    bars = {}
    for s in PAIRES:
        try:
            a = data.long_bars(s, "fx", "1H")
            a = a[((a[:, 0] // 86400 + 3) % 7) < 5]
            if len(a) > 24 * 400:
                bars[s] = a[:, :5]
        except Exception as ex:
            print("ABSENT", s, ex, flush=True)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = float(np.quantile(t_all, 0.6))
    print("ACTIFS", len(bars), "coupure", time.strftime("%d/%m/%Y", time.gmtime(cut)), flush=True)

    def st(x):
        if not x:
            return {"trades": 0}
        r = np.array([t[3] for t in x])
        return {"trades": len(r), "reussite": round(float((r > 0).mean()), 3), "gain_moyen_R": round(float(r.mean()), 3),
                "total_R": round(float(r.sum()), 1)}
    res = {}
    for g in ("swing", "h24"):
        for se in SESSIONS:
            key = g + "_" + se
            res[key] = {}
            allt = []
            for s, a in bars.items():
                tt = trades(a, s, se, g)
                allt += tt
                res[key][s] = {"apprentissage": st([t for t in tt if t[0] < cut]), "test": st([t for t in tt if t[0] >= cut]),
                               "tout": st(tt)}
            print("== GESTION", g, "SESSIONS", se, "| TOUTES PAIRES test", json.dumps(st([t for t in allt if t[0] >= cut])),
                  "| tout", json.dumps(st(allt)), flush=True)
            for s in sorted(bars, key=lambda z: -res[key][z]["tout"].get("gain_moyen_R", -9)):
                v = res[key][s]
                print("   %-7s tout %4d tr %5.1f %% %+.3fR | apprentissage %+.3fR | test %+.3fR (%d tr)" % (
                    s, v["tout"].get("trades", 0), 100 * v["tout"].get("reussite", 0), v["tout"].get("gain_moyen_R", 0),
                    v["apprentissage"].get("gain_moyen_R", 0), v["test"].get("gain_moyen_R", 0), v["test"].get("trades", 0)),
                    flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_par_paire.json", "w") as f:
        json.dump({"genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "resultats": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
