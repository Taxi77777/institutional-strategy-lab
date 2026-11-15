"""Test de l'indicateur MT4 XARDFX-38 envoye par le trader (15 nov.) : signal StopLight BUY / SELL.
   BUY  : cloture > SMMA(7) des plus hauts de la bougie precedente, LWMA(28) de la bougie precedente sous la TMA2,
          CCI(28) >= 0 ;  SELL : l'inverse.  Signal quand l'etat change, a la cloture de la bougie.
La TMA2 de l'indicateur est CENTREE : sur l'historique elle utilise les 28 bougies suivantes (elle repeint). On compare :
   - « graphique » : TMA2 centree (ce que montre l'historique, impossible en direct) ;
   - « direct »    : TMA2 calculee avec les seules bougies connues (ce que l'indicateur affiche vraiment en direct).
Trade : entree a l'ouverture suivante ; sortie au signal inverse ou au stop (2 ATR), ou objectif / stop fixes (1,5 ATR).
Test honnete : 60 % anciens / 40 % recents, frais deduits.   python -m lab.xard fx 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import COST

log = logging.getLogger("lab")
N = 28


def smma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = x[:n].mean()
    for t in range(n, len(x)):
        out[t] = (out[t - 1] * (n - 1) + x[t]) / n
    return out


def lwma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    out = np.full(len(x), np.nan)
    v = np.convolve(x, w[::-1], mode="valid") / w.sum()
    out[n - 1:] = v
    return out


def sma(x, n):
    out = np.full(len(x), np.nan)
    c = np.cumsum(np.r_[0.0, x])
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def cci(h, l, c, n=28):
    tp = (h + l + c) / 3
    m = sma(tp, n)
    md = np.full(len(tp), np.nan)
    for t in range(n - 1, len(tp)):
        md[t] = np.mean(np.abs(tp[t - n + 1:t + 1] - m[t]))
    return (tp - m) / (0.015 * np.where(md > 0, md, np.nan))


def tma2(p, centree):
    """Poids de l'indicateur : (N+1) sur la bougie, k = N..1 sur les N precedentes et, si centree, les N suivantes."""
    n = len(p)
    w = np.r_[N + 1, np.arange(N, 0, -1)].astype(float)
    out = np.full(n, np.nan)
    for t in range(N, n):
        s = w[0] * p[t] + np.dot(w[1:], p[t - N:t][::-1])
        sw = w.sum()
        if centree:
            m = min(N, n - 1 - t)
            if m > 0:
                s += np.dot(w[1:m + 1], p[t + 1:t + m + 1])
                sw += w[1:m + 1].sum()
        out[t] = s / sw
    return out


def signals(a, centree):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    sh, sl_ = smma(h, 7), smma(l, 7)
    tr = np.zeros(len(c))
    for t in range(1, len(c)):
        tr[t] = tr[t - 1]
        if c[t] > sh[t - 1]:
            tr[t] = 1
        if c[t] < sl_[t - 1]:
            tr[t] = -1
    t1 = lwma(c, N)
    t2 = tma2((h + l + 2 * c) / 4, centree)
    cc = cci(h, l, c)
    st = np.zeros(len(c))
    prev = np.r_[np.nan, t1[:-1]]
    st[(tr == 1) & (prev < t2) & (cc >= 0)] = 1
    st[(tr == -1) & (prev > t2) & (cc < 0)] = -1
    sig = np.zeros(len(c))
    ch = np.r_[False, st[1:] != st[:-1]]
    sig[ch & (st != 0)] = st[ch & (st != 0)]
    return sig


def atr(a, n=14):
    h, l, c = a[:, 2], a[:, 3], a[:, 4]
    pc = np.r_[c[0], c[:-1]]
    return sma(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))), n)


def trades(a, sig, kind, sortie="inverse", hr=None):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    A = atr(a)
    out, t, n = [], 30, len(a)
    while t < n - 2:
        d = sig[t]
        if not d or not np.isfinite(A[t]) or A[t] <= 0 or (hr is not None and not 7 <= hr[t] < 17):
            t += 1
            continue
        e = o[t + 1]
        k_sl = 2.0 if sortie == "inverse" else 1.5
        risk = k_sl * A[t]
        stop, tg = e - d * risk, (e + d * 1.5 * A[t] if sortie == "atr" else None)
        x, j = None, t + 1
        while j < min(n, t + 200):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x = stop if j == t + 1 else (min(o[j], stop) if d == 1 else max(o[j], stop))
                break
            if tg is not None and j > t + 1 and ((h[j] >= tg) if d == 1 else (l[j] <= tg)):
                x = tg
                break
            if sortie == "inverse" and sig[j] == -d and j + 1 < n:
                x, j = o[j + 1], j + 1
                break
            j += 1
        if x is None:
            x, j = c[min(j, n - 1)], min(j, n - 1)
        out.append((a[t + 1, 0], float((x - e) * d / risk - COST[kind] * e / risk), d))
        t = j
    return out


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, tf)
            if len(a) > 1500:
                bars[s] = a[-20000:]
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = np.quantile(t_all, 0.6)
    days = (t_all.max() - cut) / 86400
    print("ACTIFS", kind, tf, len(bars), flush=True)
    res = []
    sigs = {(s, ce): signals(a, ce) for s, a in bars.items() for ce in (True, False)}
    for ce in (True, False):
        for so in ("inverse", "atr"):
            tr_, te = [], []
            for s, a in bars.items():
                hr = ((a[:, 0] % 86400) // 3600).astype(int) if kind == "fx" else None
                for t0, r, d in trades(a, sigs[(s, ce)], kind, so, hr):
                    (te if t0 >= cut else tr_).append(r)

            def st(x, dd=None):
                if not x:
                    return {"trades": 0}
                r = np.array(x)
                return {"trades": len(r), "reussite": round(float((r > 0).mean()), 3),
                        "gain_moyen_R": round(float(r.mean()), 3), "total_R": round(float(r.sum()), 1)}
            v = {"tma2": "centree (graphique)" if ce else "direct (reel)", "sortie": so}
            res.append({"variante": v, "apprentissage": st(tr_), "test": st(te), "par_jour": round(len(te) / max(1, days), 2)})
            print("VARIANTE", json.dumps(v, ensure_ascii=False), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]),
                  "| TEST", json.dumps(res[-1]["test"]), "| par jour", res[-1]["par_jour"], flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/xard_%s_%s.json" % (kind, tf), "w") as f:
        json.dump({"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "variantes": res}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
